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
    use axum::{
        Router,
        extract::State,
        http::StatusCode,
        response::{IntoResponse, Response, Sse},
        routing::{get, post},
    };
    use axum::response::sse::{Event as SseEvent, KeepAlive};
    use reinfer_tokenizer::Tokenizer;
    use std::{
        net::SocketAddr,
        path::PathBuf,
        sync::{
            Arc, Mutex,
            atomic::{AtomicU64, Ordering},
        },
    };
    use tokio_stream::StreamExt as _;
    use crate::Verbosity;
    use crate::pipeline::{GenParams, generate_stream};
struct AppState {
    engine: Mutex<reinfer_cuda::engine::Engine>,
    tokenizer: Tokenizer,
    eos: Option<u32>,
    model_id: String,
    model_dir: PathBuf,
    id_seq: AtomicU64,
}

/// 同步入口（main 调）：服务阻断运行。
pub fn run_sync(args: ServeArgs, vlog: &Verbosity) -> i32 {
    if args.max_num_seqs != 1 {
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

    // CUDA 引擎装载（服务开始前完成；单实例）
    use reinfer_core::DeviceId;
    use reinfer_cuda::{CudaContext, CudaStream};
    let dev_idx = args.device.unwrap_or(0);
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
    let model_id = args.served_model_name.clone().unwrap_or_else(|| args.model.clone());
    let api_key = args.api_key.clone();

    let state = Arc::new(AppState {
        engine: Mutex::new(engine),
        tokenizer,
        eos: cfg.eos_id,
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
        app.layer(axum::middleware::from_fn(move |req: axum::extract::Request, next: axum::middleware::Next| {
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
        }))
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
        if let Ok(mut s) = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
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
async fn completion_impl(
    st: Arc<AppState>,
    body: serde_json::Value,
    chat: bool,
) -> Response {
    // ---- prompt 构造与 encode（模板渲染在阻塞前完成） ----
    let (prompt_text, parse_special) = if chat {
        let messages = match body.get("messages").and_then(|m| m.as_array()) {
            Some(a) if !a.is_empty() => a,
            _ => {
                return openai_err(
                    StatusCode::BAD_REQUEST,
                    "must provide non-empty `messages`",
                    "invalid_request_error",
                    "messages",
                    "",
                )
                .into_response();
            }
        };
        let prompt = match crate::pipeline::render_chat_template(&st.model_dir, messages) {
            Ok(Some(text)) => text,
            Ok(None) | Err(_) => {
                // 无模板/渲染失败：最后一条 content 兜底
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
                return openai_err(
                    StatusCode::BAD_REQUEST,
                    "must provide `prompt`",
                    "invalid_request_error",
                    "prompt",
                    "",
                )
                .into_response();
            }
        }
    };

    let ids = match st.tokenizer.encode(&prompt_text, parse_special) {
        Ok(v) => v,
        Err(e) => {
            return openai_err(
                StatusCode::BAD_REQUEST,
                &format!("encode: {e}"),
                "invalid_request_error",
                "prompt",
                "",
            )
            .into_response();
        }
    };

    // ---- 采样参数（OpenAI 请求体面；缺省 = OpenAI 工程表） ----
    let f = |k: &str| body.get(k).and_then(|v| v.as_f64()).map(|v| v as f32);
    let i = |k: &str| body.get(k).and_then(|v| v.as_u64()).map(|v| v as usize);
    let params = GenParams {
        temperature: f("temperature").unwrap_or(1.0),
        top_p: f("top_p").filter(|p| *p != 1.0),
        top_k: i("top_k"),
        repeat_penalty: None, // OpenAI 无该参数；服务缺省 1.0（不惩罚）
        seed: body.get("seed").and_then(|v| v.as_u64()),
    };
    let max_tokens = i("max_tokens").unwrap_or(256) as u32;
    let want_stream = body
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let eos_id = st.eos;

    let id = completion_id(&st);
    let model = st.model_id.clone();
    let prompt_tokens = ids.len();

    if want_stream {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        let st2 = Arc::clone(&st);
        let id2 = id.clone();
        let model2 = model.clone();
        let ids_c = ids.clone();
        let params_c = params.clone();
        let stream_obj = if chat { "chat.completion.chunk" } else { "text_completion.chunk" };
        let stream_obj_c = stream_obj.to_string();
        let _ = tokio::task::spawn_blocking(move || {
            let mut engine = st2.engine.lock().unwrap();
            let stat = generate_stream(
                &mut engine,
                &st2.tokenizer,
                &ids_c,
                &params_c,
                eos_id,
                max_tokens,
                |delta| {
                    let is_chat = stream_obj_c.starts_with("chat.");
                    let frame = serde_json::json!({
                        "id": id2,
                        "object": stream_obj_c,
                        "model": model2,
                        "choices": [{
                            "index": 0,
                            if is_chat { "delta" } else { "text" }: delta,
                            "finish_reason": None::<String>,
                        }],
                    })
                    .to_string();
                    tx.send(frame).is_ok()
                },
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
                            "finish_reason": if s.stopped_by_eos { "stop" } else { "length" },
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
        let stream =
            tokio_stream::wrappers::UnboundedReceiverStream::new(rx).map(|s| {
                Ok::<_, std::convert::Infallible>(SseEvent::default().data(s))
            });
        return Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response();
    }

    // stream=false：收集完整文本（closure 发裸增量串）
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    let st2 = Arc::clone(&st);
    let ids_c = ids.clone();
    let params_c = params.clone();
    let task = tokio::task::spawn_blocking(move || {
        let mut engine = st2.engine.lock().unwrap();
        generate_stream(
            &mut engine,
            &st2.tokenizer,
            &ids_c,
            &params_c,
            eos_id,
            max_tokens,
            |delta| {
                let _ = tx.send(delta.to_string());
                true
            },
        )
    });
    let stat = match task.await {
        Ok(s) => s,
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
    let choice = if chat {
        serde_json::json!({ "index": 0, "message": { "role": "assistant", "content": text },
            "finish_reason": if stat.stopped_by_eos { "stop" } else { "length" } })
    } else {
        serde_json::json!({ "index": 0, "text": text,
            "finish_reason": if stat.stopped_by_eos { "stop" } else { "length" } })
    };
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
