//! `serve`：OpenAI 兼容 HTTP 服务（契约 v2.1/v2.5；axum）。
//!
//! 面：`GET /v1/models` · `GET /healthz` · `POST /v1/chat/completions`
//! （含 `stream: true` SSE）· `POST /v1/completions`；`--api-key`（Bearer）鉴权；
//! graceful shutdown（SIGINT/SIGTERM）。日志一律 stderr；stdout 无输出。
//!
//! V1 并发语义：模型单实例 + `max-num-seqs=1`（请求队列串行——引擎每请求
//! 独立 KV，无跨请求缓存；流式不受影响）。`stop`/`logprobs` 等参数
//! 解析接受但暂不生效（记录；后续面）。

use std::str::FromStr; // ServeArgs value_parser = Backend::from_str（clap derive 展开于本文件头）

/// serve 参数（契约 v2.1 面）。
#[derive(clap::Args, Debug)]
pub struct ServeArgs {
    /// Model repo (owner/model) or local model directory
    pub model: String,

    /// Backend (auto|cuda|ascend|cpu)
    #[arg(long = "backend", value_name = "BACKEND", value_parser = crate::Backend::from_str, default_value = "auto")]
    pub backend: crate::Backend,

    /// Compute device index (0-based; CUDA view)
    #[arg(long = "device", value_name = "ID")]
    pub device: Option<u32>,

    /// Listen host
    #[arg(long = "host", value_name = "HOST", default_value = "0.0.0.0")]
    pub host: String,

    /// Listen port
    #[arg(long = "port", value_name = "PORT", default_value_t = 8000)]
    pub port: u16,

    /// Effective context window (defaults to model max_position_embeddings)
    #[arg(long = "max-model-len", value_name = "N")]
    pub max_model_len: Option<usize>,

    /// Max concurrent sequences (V1: only 1 supported)
    #[arg(long = "max-num-seqs", value_name = "N", default_value_t = 1)]
    pub max_num_seqs: usize,

    /// Model name exposed via /v1/models and echoed in responses
    #[arg(long = "served-model-name", value_name = "NAME")]
    pub served_model_name: Option<String>,

    /// Bearer token required on /v1/* (when set)
    #[arg(long = "api-key", value_name = "KEY")]
    pub api_key: Option<String>,

    /// Prometheus metrics endpoint (not implemented in P1)
    #[arg(long = "metrics")]
    pub metrics: bool,
}

#[cfg(feature = "cuda")]
mod backend {
    use super::*;
    use crate::Verbosity;
    use crate::pipeline::{GenParams, generate_stream};
    use crate::sched_loop::{SchedFrame, SubmitRequest};
    use axum::response::sse::{Event as SseEvent, KeepAlive};
    use axum::{
        Router,
        extract::State,
        http::StatusCode,
        response::{IntoResponse, Response, Sse},
        routing::{get, post},
    };
    use reinfer_tokenizer::Tokenizer;
    use std::{
        net::SocketAddr,
        path::{Path, PathBuf},
        sync::{
            Arc, Mutex,
            atomic::{AtomicU64, Ordering},
        },
    };
    use tokio_stream::StreamExt as _;
    struct AppState {
        /// Serial-path engine (REINFER_SCHEDULER=off); None on the
        /// scheduler path (the loop thread owns the device).
        engine: Option<Mutex<reinfer_cuda::engine::Engine>>,
        /// S2-D scheduler loop handle (REINFER_SCHEDULER=on); None on the
        /// serial path.
        sched: Option<Arc<crate::sched_loop::SchedHandle>>,
        tokenizer: Arc<Tokenizer>,
        eos: Option<u32>,
        model_id: String,
        max_len: usize,
        model_dir: PathBuf,
        id_seq: AtomicU64,
    }

    /// 同步入口（main 调）：服务阻断运行。
    pub fn run_sync(args: ServeArgs, vlog: &Verbosity) -> i32 {
        // S2-D: REINFER_SCHEDULER=on routes requests through the scheduler
        // loop, which owns the device — max-num-seqs > 1 becomes meaningful
        // (admission caps concurrency). The serial path stays max-num-seqs=1.
        let sched_on = crate::sched_loop::scheduler_env_on();
        if !sched_on && args.max_num_seqs != 1 {
            eprintln!(
                "reinfer: serve: V1 仅支持 --max-num-seqs=1（串行），收到 {}",
                args.max_num_seqs
            );
            return 2;
        }
        if args.backend == crate::Backend::Ascend || args.backend == crate::Backend::Cpu {
            eprintln!("reinfer: serve: backend not implemented yet (cuda only)");
            return 2;
        }
        if args.metrics {
            eprintln!("reinfer: serve: --metrics not implemented in P1; ignored");
        }

        let dir = match crate::resolve_model_dir(&args.model) {
            Ok(d) => d,
            Err(()) => return 2,
        };

        // config → LlamaConfig（校验 + ctx 缺省）
        let cfg_val: serde_json::Value = match std::fs::read(dir.join("config.json"))
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
        {
            Some(v) => v,
            None => {
                eprintln!("reinfer: serve: config.json missing in {dir:?}");
                return 2;
            }
        };
        let cfg = match reinfer_arch::llama::from_hf_config(&cfg_val) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("reinfer: serve: config.json: {e}");
                return 2;
            }
        };
        let max_len = args.max_model_len.unwrap_or(cfg.ctx_len);

        // tokenizer
        let tok_val: serde_json::Value = match std::fs::read(dir.join("tokenizer.json"))
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
        {
            Some(v) => v,
            None => {
                eprintln!("reinfer: serve: tokenizer.json missing");
                return 2;
            }
        };
        let tcfg_val: serde_json::Value = std::fs::read(dir.join("tokenizer_config.json"))
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or(serde_json::Value::Null);
        let tokenizer = match Tokenizer::from_hf_json(&tok_val, &tcfg_val) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("reinfer: serve: tokenizer: {e}");
                return 2;
            }
        };
        // Tokenizer 不可 Clone（S2-D 需跨线程 share）→ Arc（auto-deref，调用面不变）
        let tokenizer = Arc::new(tokenizer);

        // CUDA 引擎装载（服务开始前完成；单实例）。S2-D：scheduler 开启时
        // 设备由循环线程持有（context + engine + 共享 KV 池），主线程不碰。
        use reinfer_core::DeviceId;
        let dev_idx = args.device.unwrap_or(0);
        let (engine_slot, sched_slot) = if sched_on {
            // S2-D: derive the shared KV pool budget, then spawn the loop —
            // the init closure runs CudaContext::init + engine load + pool
            // alloc (incl. the anchor window) on the loop thread, blocking,
            // so any init failure surfaces here before we listen.
            let kv_pages = match sched_kv_pages(&dir, &cfg, max_len, dev_idx) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("reinfer: serve: scheduler KV budget: {e}");
                    return 2;
                }
            };
            let sched_cfg = crate::sched_loop::SchedLoopConfig {
                base_seed: crate::sched_loop::base_seed_env(),
                vocab: cfg.vocab_size,
                dev: dev_idx,
                n_layer: cfg.n_layer,
                block_len: crate::sched_loop::BLOCK_LEN,
                max_model_len: max_len,
                kv_pages,
                max_num_seqs: args.max_num_seqs,
                chunk_size: max_len, // V1: single-chunk prefill per request
                max_steps: 0,
                detok: {
                    let tok = Arc::clone(&tokenizer);
                    Arc::new(move |ids: &[u32]| tok.decode_all(ids))
                },
                // 016 r2: prefix cache (P3-01 v1) — on by default;
                // REINFER_PREFIX_CACHE_PAGES overrides the 10%-of-pool budget.
                prefix_cache_pages: crate::sched_loop::prefix_cache_pages_env(
                    kv_pages,
                    crate::sched_loop::prefix_cache_env_on(),
                ),
            };
            let window = sched_cfg.window_pages();
            let admit_cap = sched_cfg.admit_cap();
            let dir_c = dir.clone();
            let handle = match crate::sched_loop::SchedHandle::spawn(move || {
                crate::sched_loop::CudaBatchExecutor::load(dev_idx, &dir_c, max_len, kv_pages)
                    .map(|exec| (exec, sched_cfg))
            }) {
                Ok(h) => h,
                Err(e) => {
                    eprintln!("reinfer: serve: scheduler spawn: {e}");
                    return 2;
                }
            };
            if vlog.at(1) {
                eprintln!(
                    "reinfer: serve: scheduler on (REINFER_SCHEDULER): kv_pages={kv_pages}, window={window}, admit_cap={admit_cap}"
                );
            }
            (None, Some(Arc::new(handle)))
        } else {
            use reinfer_cuda::{CudaContext, CudaStream};
            let ctx = match CudaContext::init(DeviceId::new(dev_idx)) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("reinfer: serve: cuda init (device {dev_idx}): {e}");
                    return 2;
                }
            };
            let _stream = CudaStream::new(ctx.device_id()).expect("stream");
            let arch = reinfer_cuda::arch::resolve_arch().expect("arch");
            let engine = match reinfer_cuda::engine::Engine::load(
                ctx.device_id().clone(),
                &arch,
                Some(std::env::temp_dir().join("reinfer-jit-dense")),
                &dir,
                max_len,
            ) {
                Ok(e) => e,
                Err(e) => {
                    eprintln!("reinfer: serve: engine load: {e}");
                    return 2;
                }
            };
            if vlog.at(1) {
                eprintln!("reinfer: serve: engine loaded (arch={arch}, ctx={max_len})");
            }
            (Some(Mutex::new(engine)), None)
        };
        let model_id = args.served_model_name.clone().unwrap_or_else(|| args.model.clone());
        let api_key = args.api_key.clone();

        // EOS 优先级：generation_config.json > config.json > tokenizer eos（014 D8）
        let gen_cfg: serde_json::Value = std::fs::read(dir.join("generation_config.json"))
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or(serde_json::Value::Null);
        let eos = reinfer_arch::llama::resolve_eos(&cfg_val, Some(&gen_cfg), tokenizer.eos_token());
        if vlog.at(1) {
            eprintln!("reinfer: serve: eos resolved = {eos:?}");
        }

        let state = Arc::new(AppState {
            engine: engine_slot,
            sched: sched_slot,
            tokenizer,
            eos,
            max_len,
            model_id,
            model_dir: dir,
            id_seq: AtomicU64::new(0),
        });

        let app = Router::new()
            .route("/healthz", get(healthz))
            .route("/v1/models", get(list_models))
            .route("/v1/chat/completions", post(chat_completions))
            .route("/v1/completions", post(completions))
            .with_state(state);
        let app = if let Some(key) = api_key {
            app.layer(axum::middleware::from_fn(
                move |req: axum::extract::Request, next: axum::middleware::Next| {
                    let key = key.clone();
                    async move {
                        let authed = req
                            .headers()
                            .get(axum::http::header::AUTHORIZATION)
                            .and_then(|v| v.to_str().ok())
                            .and_then(|v| v.strip_prefix("Bearer "))
                            .map(|t| t == key)
                            .unwrap_or(false);
                        if !authed {
                            return openai_err(
                                StatusCode::UNAUTHORIZED,
                                "invalid or missing API key",
                                "invalid_request_error",
                                "api_key",
                                "unauthorized",
                            )
                            .into_response();
                        }
                        next.run(req).await
                    }
                },
            ))
        } else {
            app
        };

        let addr: SocketAddr = match format!("{}:{}", args.host, args.port).parse() {
            Ok(a) => a,
            Err(_) => {
                eprintln!("reinfer: serve: invalid addr {}:{}", args.host, args.port);
                return 2;
            }
        };
        eprintln!("reinfer: serve: listening on http://{addr} (model {})", args.model);

        let rt = match tokio::runtime::Runtime::new() {
            Ok(rt) => rt,
            Err(e) => {
                eprintln!("reinfer: serve: tokio runtime: {e}");
                return 2;
            }
        };
        rt.block_on(async move {
            let listener = match tokio::net::TcpListener::bind(addr).await {
                Ok(l) => l,
                Err(e) => {
                    eprintln!("reinfer: serve: bind {addr}: {e}");
                    std::process::exit(4);
                }
            };
            tokio::select! {
                r = axum::serve(listener, app) => {
                    if let Err(e) = r {
                        eprintln!("reinfer: serve: server error: {e}");
                        std::process::exit(3);
                    }
                }
                _ = shutdown_signal() => {
                    eprintln!("reinfer: serve: shutdown signal received");
                }
            }
        });
        0
    }

    /// graceful shutdown（SIGINT/SIGTERM）。
    async fn shutdown_signal() {
        let ctrl_c = async {
            let _ = tokio::signal::ctrl_c().await;
        };
        #[cfg(unix)]
        let terminate = async {
            if let Ok(mut s) =
                tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            {
                let _ = s.recv().await;
            }
        };
        #[cfg(not(unix))]
        let terminate = std::future::pending::<()>();
        tokio::select! {
            _ = ctrl_c => {},
            _ = terminate => {},
        }
    }

    /// S2-D：调度器共享 KV 池预算。镜像 vLLM gpu_memory_utilization=0.9：
    /// 设备显存 − 权重 − 引擎 singleton 池（Engine::load 恒分配一整窗的
    /// per-layer KV 页），按 90% 利用率建池；不足一个窗口即拒绝启动。
    ///
    /// ## 页口径换算（2026-09-01 修复）
    ///
    /// `kv_budget_pages` 是 vLLM 口径：页 = 一个 block_len 块、**跨所有层**
    /// （`page_bytes_f16`），窗口 = `ceil(max_len/block_len)` 块。而
    /// `CudaBatchExecutor`/`KvStore`/`KvSegmentPool` 用引擎口径：页 =
    /// **单层**一 block（页表 `li*pp + j`），窗口 = `n_layer × pp` 页。
    /// 此前两者混用（misc 虚高 28×、判据无量纲、返回值少 28×）→ 预算
    /// 2169 < 窗 3584 假报错。此处显式换算：blocks（预算）↔ per-layer
    /// pages（executor），数值由几何恒等 `pp × page_bytes_f16 ==
    /// n_layer×pp × per-layer-page-bytes` 保证一致。
    fn sched_kv_pages(
        dir: &Path,
        cfg: &reinfer_arch::llama::LlamaConfig,
        max_len: usize,
        dev_idx: u32,
    ) -> Result<usize, String> {
        use reinfer_memory::budget::{KvBudgetInput, KvGeometry, kv_budget_pages};
        let info = reinfer_cuda::CudaContext::device_info(dev_idx)
            .map_err(|e| format!("device info (dev {dev_idx}): {e}"))?;
        let geom = KvGeometry {
            n_layer: cfg.n_layer,
            kv_heads: cfg.kv_heads,
            head_dim: cfg.head_dim,
            block_len: crate::sched_loop::BLOCK_LEN,
        };
        let pp = max_len.div_ceil(geom.block_len); // blocks per window (跨层口径)
        // 引擎 singleton 池（Engine::load 恒分配 n_layer×pp per-layer 页）的
        // 字节占用 = pp × page_bytes_f16（两者几何恒等，见上文）。
        let misc_bytes = geom.n_layer as u64
            * pp as u64
            * (geom.block_len * geom.kv_heads * geom.head_dim * 2 * 2) as u64; // per-layer 页 × 单层页字节（K+V）
        let input = KvBudgetInput {
            mem_total_bytes: info.total_mem,
            weights_bytes: dir_bytes(dir),
            graph_pool_bytes: 0, // graph pool grows lazily; utilization headroom covers it
            misc_bytes,
            utilization: 0.9,
        };
        let pages_blocks = kv_budget_pages(&input, &geom) as usize;
        if pages_blocks < pp {
            return Err(format!(
                "KV budget {pages_blocks} blocks < one window ({pp}) — reduce --max-model-len or free device memory"
            ));
        }
        // 换算为 executor per-layer 页数（KvStore/KvSegmentPool/锚段口径）。
        Ok(pages_blocks * geom.n_layer)
    }

    /// 模型目录文件总字节（权重在设备侧的占用）。
    fn dir_bytes(dir: &Path) -> u64 {
        let Ok(rd) = std::fs::read_dir(dir) else { return 0 };
        rd.flatten()
            .map(|e| {
                if e.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                    dir_bytes(&e.path())
                } else {
                    e.metadata().map(|m| m.len()).unwrap_or(0)
                }
            })
            .sum()
    }

    // ---------------- handlers ----------------

    async fn healthz() -> axum::Json<serde_json::Value> {
        axum::Json(serde_json::json!({ "status": "ok" }))
    }

    async fn list_models(State(st): State<Arc<AppState>>) -> axum::Json<serde_json::Value> {
        axum::Json(serde_json::json!({
            "object": "list",
            "data": [{
                "id": st.model_id,
                "object": "model",
                "created": 0,
                "owned_by": "reinfer",
            }]
        }))
    }

    #[axum::debug_handler]
    async fn chat_completions(
        State(st): State<Arc<AppState>>,
        axum::extract::Json(body): axum::extract::Json<serde_json::Value>,
    ) -> Response {
        completion_impl(Arc::clone(&st), body, true).await
    }

    async fn completions(
        State(st): State<Arc<AppState>>,
        axum::extract::Json(body): axum::extract::Json<serde_json::Value>,
    ) -> Response {
        completion_impl(Arc::clone(&st), body, false).await
    }

    /// 统一完成端（chat=true：messages → chat 模板渲染；否则 prompt 文本直编码）。
    /// S2-D：REINFER_SCHEDULER=on 时交调度器循环（sched_completion），否则走
    /// 原串行单请求路径（max-num-seqs=1，行为不变）。
    async fn completion_impl(st: Arc<AppState>, body: serde_json::Value, chat: bool) -> Response {
        if st.sched.is_some() {
            return sched_completion(&st, body, chat).await;
        }
        let parsed = match parse_completion(&st, body, chat) {
            Ok(p) => p,
            Err(r) => return r,
        };
        let CompletionReq {
            ids,
            params,
            max_tokens,
            lp_top_n,
            want_stream,
            id,
            model,
            prompt_tokens,
            stop,
        } = parsed;
        // S3-1 串行路径 stop：尚未实现（generate_stream 无 stop 参数）——
        // 记为"后续面"（T7 stop 经调度器面验证）；此处显式接住避免静默忽略。
        let _ = stop;
        let eos_id = st.eos;
        let want_logprobs = lp_top_n > 0;

        if want_stream {
            let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<String>();
            let st2 = Arc::clone(&st);
            let id2 = id.clone();
            let model2 = model.clone();
            let ids_c = ids.clone();
            let params_c = params.clone();
            let stream_obj = if chat { "chat.completion.chunk" } else { "text_completion.chunk" };
            let stream_obj_c = stream_obj.to_string();
            let lp_off = lp_top_n;
            let _ = tokio::task::spawn_blocking(move || {
                let mut engine = st2
                    .engine
                    .as_ref()
                    .expect("serial engine (REINFER_SCHEDULER off)")
                    .lock()
                    .unwrap();
                let stat = generate_stream(
                    &mut engine,
                    &st2.tokenizer,
                    &ids_c,
                    &params_c,
                    eos_id,
                    &stop,
                    max_tokens,
                    |delta, tok| {
                        let is_chat = stream_obj_c.starts_with("chat.");
                        let mut choice = if is_chat {
                            serde_json::json!({ "index": 0, "delta": { "content": delta }, "finish_reason": None::<String> })
                        } else {
                            serde_json::json!({ "index": 0, "text": delta, "finish_reason": None::<String> })
                        };
                        if lp_off > 0
                            && let Some(o) = tok
                        {
                            let lp_val = serde_json::json!({ "content": [lp_json(o, lp_off, &st2.tokenizer)] });
                            let ch = choice.as_object_mut().unwrap();
                            if is_chat {
                                ch["delta"]["logprobs"] = lp_val;
                            } else {
                                ch["logprobs"] = lp_val;
                            }
                        }
                        let frame = serde_json::json!({
                            "id": id2,
                            "object": stream_obj_c,
                            "model": model2,
                            "choices": [choice],
                        })
                        .to_string();
                        tx.send(frame).is_ok()
                    },
                    lp_off,
                );
                match stat {
                    Ok(s) => {
                        let is_chat = stream_obj_c.starts_with("chat.");
                        let finish = serde_json::json!({
                            "id": id2,
                            "object": stream_obj_c,
                            "model": model2,
                            "choices": [{
                                "index": 0,
                                if is_chat { "delta" } else { "text" }: {},
                                "finish_reason": if s.stopped_by_eos || s.stopped_by_stop { "stop" } else { "length" },
                            }],
                            "usage": {
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": s.tokens,
                                "total_tokens": prompt_tokens + s.tokens,
                            },
                        })
                        .to_string();
                        let _ = tx.send(finish);
                        let _ = tx.send("[DONE]".to_string());
                    }
                    Err(e) => {
                        let err = serde_json::json!({
                        "error": { "message": e, "type": "server_error", "param": "", "code": "generation_failed" },
                    })
                    .to_string();
                        let _ = tx.send(err);
                        let _ = tx.send("[DONE]".to_string());
                    }
                }
            });
            let stream = tokio_stream::wrappers::UnboundedReceiverStream::new(rx)
                .map(|s| Ok::<_, std::convert::Infallible>(SseEvent::default().data(s)));
            return Sse::new(stream).keep_alive(KeepAlive::default()).into_response();
        }

        // stream=false：收集完整文本（closure 发裸增量串）
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        let st2 = Arc::clone(&st);
        let ids_c = ids.clone();
        let params_c = params.clone();
        let task = tokio::task::spawn_blocking(move || {
            let mut engine =
                st2.engine.as_ref().expect("serial engine (REINFER_SCHEDULER off)").lock().unwrap();
            let mut det: Vec<crate::pipeline::TokenOut> = Vec::new();
            let stat = generate_stream(
                &mut engine,
                &st2.tokenizer,
                &ids_c,
                &params_c,
                eos_id,
                &stop,
                max_tokens,
                |delta, tok| {
                    let _ = tx.send(delta.to_string());
                    if let Some(o) = tok {
                        det.push(o.clone());
                    }
                    true
                },
                lp_top_n,
            );
            (stat, det)
        });
        let (stat, det) = match task.await {
            Ok(v) => v,
            Err(e) => {
                return openai_err(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    &format!("generation task: {e}"),
                    "server_error",
                    "",
                    "generation_failed",
                )
                .into_response();
            }
        };
        let stat = match stat {
            Ok(s) => s,
            Err(e) => {
                return openai_err(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    &e,
                    "server_error",
                    "",
                    "generation_failed",
                )
                .into_response();
            }
        };
        let mut text = String::new();
        while let Ok(chunk) = rx.try_recv() {
            text.push_str(&chunk);
        }
        let object = if chat { "chat.completion" } else { "text_completion" };
        let mut choice = if chat {
            serde_json::json!({ "index": 0, "message": { "role": "assistant", "content": text },
            "finish_reason": if stat.stopped_by_eos { "stop" } else { "length" } })
        } else {
            serde_json::json!({ "index": 0, "text": text,
            "finish_reason": if stat.stopped_by_eos { "stop" } else { "length" } })
        };
        if want_logprobs {
            choice["logprobs"] = serde_json::json!({
                "content": det.iter().map(|o| lp_json(o, lp_top_n, &st.tokenizer)).collect::<Vec<_>>()
            });
        }
        let payload = serde_json::json!({
            "id": id,
            "object": object,
            "created": now_unix(),
            "model": model,
            "choices": [choice],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": stat.tokens,
                "total_tokens": prompt_tokens + stat.tokens,
            },
        });
        axum::Json(payload).into_response()
    }

    /// 请求体 → 生成请求（serial 与 scheduler 路径共用解析；错误即 OpenAI 面）。
    fn parse_completion(
        st: &AppState,
        body: serde_json::Value,
        chat: bool,
    ) -> Result<CompletionReq, Response> {
        // ---- prompt 构造与 encode（模板渲染在阻塞前完成） ----
        let (prompt_text, parse_special) = if chat {
            let messages = match body.get("messages").and_then(|m| m.as_array()) {
                Some(a) if !a.is_empty() => a,
                _ => {
                    return Err(openai_err(
                        StatusCode::BAD_REQUEST,
                        "must provide non-empty `messages`",
                        "invalid_request_error",
                        "messages",
                        "",
                    ));
                }
            };
            let prompt = match crate::pipeline::render_chat_template(&st.model_dir, messages) {
                Ok(Some(text)) => text,
                Ok(None) => messages
                    .last()
                    .and_then(|m| m.get("content").and_then(|c| c.as_str()))
                    .unwrap_or("")
                    .to_string(),
                Err(e) => {
                    eprintln!(
                        "reinfer: serve: chat template render failed ({e}); using raw content"
                    );
                    messages
                        .last()
                        .and_then(|m| m.get("content").and_then(|c| c.as_str()))
                        .unwrap_or("")
                        .to_string()
                }
            };
            (prompt, true)
        } else {
            match body.get("prompt").and_then(|p| p.as_str()) {
                Some(p) => (p.to_string(), false),
                None => {
                    return Err(openai_err(
                        StatusCode::BAD_REQUEST,
                        "must provide `prompt`",
                        "invalid_request_error",
                        "prompt",
                        "",
                    ));
                }
            }
        };

        let ids = match st.tokenizer.encode(&prompt_text, parse_special) {
            Ok(v) => v,
            Err(e) => {
                return Err(openai_err(
                    StatusCode::BAD_REQUEST,
                    &format!("encode: {e}"),
                    "invalid_request_error",
                    "prompt",
                    "",
                ));
            }
        };

        // ---- 采样参数（OpenAI 请求体面；缺省 = OpenAI 工程表） ----
        let f = |k: &str| body.get(k).and_then(|v| v.as_f64()).map(|v| v as f32);
        let i = |k: &str| body.get(k).and_then(|v| v.as_u64()).map(|v| v as usize);
        let params = GenParams {
            temperature: f("temperature").unwrap_or(1.0),
            top_p: f("top_p").filter(|p| *p != 1.0),
            top_k: i("top_k"),
            // S3-2: OpenAI penalty surfaces (D5 chain fields; None = off).
            repeat_penalty: None, // OpenAI 无该参数；服务缺省 1.0（不惩罚）
            frequency_penalty: f("frequency_penalty"),
            presence_penalty: f("presence_penalty"),
            seed: body.get("seed").and_then(|v| v.as_u64()),
        };
        let max_tokens = i("max_tokens").unwrap_or(256) as u32;
        if max_tokens == 0 {
            return Err(openai_err(
                StatusCode::BAD_REQUEST,
                "max_tokens must be > 0",
                "invalid_request_error",
                "max_tokens",
                "",
            ));
        }
        if params.temperature < 0.0 || params.temperature > 2.0 {
            return Err(openai_err(
                StatusCode::BAD_REQUEST,
                "temperature must be in [0, 2]",
                "invalid_request_error",
                "temperature",
                "",
            ));
        }
        if let Some(p) = params.top_p
            && (p < 0.0 || p > 1.0)
        {
            return Err(openai_err(
                StatusCode::BAD_REQUEST,
                "top_p must be in [0, 1]",
                "invalid_request_error",
                "top_p",
                "",
            ));
        }
        if ids.len() + max_tokens as usize > st.max_len {
            return Err(openai_err(
                StatusCode::BAD_REQUEST,
                &format!(
                    "maximum context length exceeded: prompt_tokens={} + max_tokens={} > ctx={}",
                    ids.len(),
                    max_tokens,
                    st.max_len
                ),
                "invalid_request_error",
                "context_length",
                "context_length_exceeded",
            ));
        }
        // logprobs 面（OpenAI 语义）：logprobs=true → 每 token 主 logp；top_logprobs → top-k
        let want_logprobs = body.get("logprobs").and_then(|v| v.as_bool()).unwrap_or(false);
        let top_logprobs = i("top_logprobs").unwrap_or(0).min(5);
        let lp_top_n = if want_logprobs { top_logprobs.max(1) } else { 0 };
        let want_stream = body.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);

        let id = completion_id(st);
        let model = st.model_id.clone();
        let prompt_tokens = ids.len();

        // S3-1: `stop`（OpenAI 面——string 或 string 数组；tokenize 为
        // 原始 id 序列（无 special），交给调度器/串行路径做增量匹配。
        // 每条 stop 串 ≤64 字符（OpenAI 参考上限），超数组 32 拒绝。
        let stop: Vec<Vec<u32>> = match body.get("stop") {
            Some(serde_json::Value::String(s)) => {
                vec![tokenize_stop(st, s)?]
            }
            Some(serde_json::Value::Array(a)) => {
                if a.len() > 32 {
                    return Err(openai_err(
                        StatusCode::BAD_REQUEST,
                        "up to 32 stop sequences allowed",
                        "invalid_request_error",
                        "stop",
                        "",
                    ));
                }
                let mut out = Vec::with_capacity(a.len());
                for v in a {
                    let s = match v.as_str() {
                        Some(s) => s,
                        None => {
                            return Err(openai_err(
                                StatusCode::BAD_REQUEST,
                                "stop entries must be strings",
                                "invalid_request_error",
                                "stop",
                                "",
                            ));
                        }
                    };
                    out.push(tokenize_stop(st, s)?);
                }
                out
            }
            _ => Vec::new(),
        };

        Ok(CompletionReq {
            ids,
            params,
            max_tokens,
            lp_top_n,
            want_stream,
            id,
            model,
            prompt_tokens,
            stop,
        })
    }

    /// Tokenize one OpenAI `stop` string (no special tokens; empty after
    /// encode → 400, mirroring vLLM's empty-stop rejection).
    fn tokenize_stop(st: &AppState, s: &str) -> Result<Vec<u32>, axum::response::Response> {
        let ids = st.tokenizer.encode(s, false).map_err(|e| {
            openai_err(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("stop encode: {e}"),
                "server_error",
                "",
                "stop",
            )
            .into_response()
        })?;
        if ids.is_empty() {
            return Err(openai_err(
                StatusCode::BAD_REQUEST,
                "stop sequence must tokenize to at least one token",
                "invalid_request_error",
                "stop",
                "",
            )
            .into_response());
        }
        Ok(ids)
    }

    /// 解析后的完成请求（serial 与 scheduler 共用面）。
    struct CompletionReq {
        ids: Vec<u32>,
        params: GenParams,
        max_tokens: u32,
        lp_top_n: usize,
        want_stream: bool,
        id: String,
        model: String,
        prompt_tokens: usize,
        /// S3-1: OpenAI `stop` sequences (token IDs; incremental matching
        /// happens in the scheduler / serial path).
        stop: Vec<Vec<u32>>,
    }

    /// S2-D：scheduler 路径（REINFER_SCHEDULER=on）。submit 到调度循环，帧经
    /// 有界 channel（256）回流 → SSE（流式）或聚合（非流式）。断连（客户端
    /// drop receiver）→ 循环侧 blocking_send 失败 → 仅 abort 该请求——无需
    /// 额外 disconnect watcher；共享池与其它请求互不污染。
    async fn sched_completion(st: &AppState, body: serde_json::Value, chat: bool) -> Response {
        let parsed = match parse_completion(st, body, chat) {
            Ok(p) => p,
            Err(r) => return r,
        };
        let handle = match st.sched.as_ref() {
            Some(h) => Arc::clone(h),
            None => {
                return openai_err(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "scheduler not enabled",
                    "server_error",
                    "",
                    "scheduler_off",
                )
                .into_response();
            }
        };
        let token = st.id_seq.fetch_add(1, Ordering::Relaxed);
        let (tx, mut rx) = tokio::sync::mpsc::channel::<SchedFrame>(256);
        if let Err(e) = handle.submit(SubmitRequest {
            ids: parsed.ids,
            params: parsed.params,
            eos: st.eos,
            max_tokens: parsed.max_tokens as usize,
            stop: parsed.stop,
            logprobs_top_n: parsed.lp_top_n,
            token,
            tx,
        }) {
            return openai_err(
                StatusCode::INTERNAL_SERVER_ERROR,
                &e,
                "server_error",
                "",
                "generation_failed",
            )
            .into_response();
        }
        let object = if chat { "chat.completion" } else { "text_completion" };
        let stream_obj = if chat { "chat.completion.chunk" } else { "text_completion.chunk" };
        let prompt_tokens = parsed.prompt_tokens;
        let lp_off = parsed.lp_top_n;
        if parsed.want_stream {
            let id = parsed.id;
            let model = parsed.model;
            let is_chat = chat;
            let tok = Arc::clone(&st.tokenizer);
            let frames = tokio_stream::wrappers::ReceiverStream::new(rx).map(move |frame| {
                let data = match frame {
                    SchedFrame::Token { delta, out } => {
                        let mut choice = if is_chat {
                            serde_json::json!({ "index": 0, "delta": { "content": delta }, "finish_reason": None::<String> })
                        } else {
                            serde_json::json!({ "index": 0, "text": delta, "finish_reason": None::<String> })
                        };
                        if lp_off > 0
                            && let Some(o) = out
                        {
                            let lp_val = serde_json::json!({ "content": [lp_json(&o, lp_off, &tok)] });
                            let ch = choice.as_object_mut().unwrap();
                            if is_chat {
                                ch["delta"]["logprobs"] = lp_val;
                            } else {
                                ch["logprobs"] = lp_val;
                            }
                        }
                        serde_json::json!({
                            "id": id,
                            "object": stream_obj,
                            "model": model,
                            "choices": [choice],
                        })
                        .to_string()
                    }
                    SchedFrame::Done { stopped_by_eos, stopped_by_stop, tokens, prompt_tokens: pt } => {
                        serde_json::json!({
                            "id": id,
                            "object": stream_obj,
                            "model": model,
                            "choices": [{
                                "index": 0,
                                if is_chat { "delta" } else { "text" }: {},
                                "finish_reason": if stopped_by_eos || stopped_by_stop { "stop" } else { "length" },
                            }],
                            "usage": {
                                "prompt_tokens": pt,
                                "completion_tokens": tokens,
                                "total_tokens": pt + tokens,
                            },
                        })
                        .to_string()
                    }
                    SchedFrame::Error { message } => serde_json::json!({
                        "error": { "message": message, "type": "server_error", "param": "", "code": "generation_failed" },
                    })
                    .to_string(),
                };
                Ok::<_, std::convert::Infallible>(SseEvent::default().data(data))
            });
            // 循环在 Done/Error 帧后必 drop sender（终止即清场）→ 通道 EOF 后再补
            // "[DONE]"（与串行路径的 finish + [DONE] 顺序一致）。
            let stream = frames.chain(tokio_stream::once(Ok::<_, std::convert::Infallible>(
                SseEvent::default().data("[DONE]"),
            )));
            return Sse::new(stream).keep_alive(KeepAlive::default()).into_response();
        }

        // stream=false：聚合帧（Token→文本；Done→usage/终止；Error→500）。
        let mut text = String::new();
        let mut det: Vec<crate::pipeline::TokenOut> = Vec::new();
        let mut finish: Option<(bool, usize)> = None;
        let mut err: Option<String> = None;
        while let Some(frame) = rx.recv().await {
            match frame {
                SchedFrame::Token { delta, out } => {
                    text.push_str(&delta);
                    if let Some(o) = out {
                        det.push(o);
                    }
                }
                SchedFrame::Done { stopped_by_eos, stopped_by_stop, tokens, .. } => {
                    finish = Some((stopped_by_eos || stopped_by_stop, tokens));
                    break;
                }
                SchedFrame::Error { message } => {
                    err = Some(message);
                    break;
                }
            }
        }
        if let Some(message) = err {
            return openai_err(
                StatusCode::INTERNAL_SERVER_ERROR,
                &message,
                "server_error",
                "",
                "generation_failed",
            )
            .into_response();
        }
        let (stopped_by_eos, completion_tokens) = finish.unwrap_or((false, 0));
        let mut choice = if chat {
            serde_json::json!({ "index": 0, "message": { "role": "assistant", "content": text },
            "finish_reason": if stopped_by_eos { "stop" } else { "length" } })
        } else {
            serde_json::json!({ "index": 0, "text": text,
            "finish_reason": if stopped_by_eos { "stop" } else { "length" } })
        };
        if lp_off > 0 {
            choice["logprobs"] = serde_json::json!({
                "content": det.iter().map(|o| lp_json(o, lp_off, &st.tokenizer)).collect::<Vec<_>>()
            });
        }
        let payload = serde_json::json!({
            "id": parsed.id,
            "object": object,
            "created": now_unix(),
            "model": parsed.model,
            "choices": [choice],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        });
        axum::Json(payload).into_response()
    }

    /// TokenOut → OpenAI `logprobs.content` 元素（非流式与流式共用）。
    /// token/top_logprobs 的 token 字段 = 解码文本（OpenAI 规范；与 vLLM decode 文本同构比较）。
    fn lp_json(o: &crate::pipeline::TokenOut, top_n: usize, tok: &Tokenizer) -> serde_json::Value {
        let token_str = tok.decode_all(&[o.token]);
        serde_json::json!({
            "token": token_str,
            "logprob": o.logprob,
            "top_logprobs": o.top.iter().take(top_n).map(|(t, l)| {
                serde_json::json!({ "token": tok.decode_all(&[*t]), "logprob": l })
            }).collect::<Vec<_>>(),
        })
    }

    fn completion_id(st: &AppState) -> String {
        let n = st.id_seq.fetch_add(1, Ordering::Relaxed);
        format!("chatcmpl-reinfer-{n}")
    }

    fn now_unix() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    }

    fn openai_err(
        status: StatusCode,
        message: &str,
        typ: &str,
        param: &str,
        code: impl Into<String>,
    ) -> axum::response::Response {
        let code: String = code.into();
        (
            status,
            axum::Json(serde_json::json!({
                "error": {
                    "message": message,
                    "type": typ,
                    "param": param,
                    "code": code,
                }
            })),
        )
            .into_response()
    }
}
#[cfg(feature = "cuda")]
pub use backend::run_sync;
