//! 生成管线：引擎 + tokenizer + 采样 + EOS/-n 语义 + 流式回调（run/serve 共用）。
//!
//! 依赖 reinfer-cuda（Engine/argmax）——整个模块按 `cuda` feature 门控。

#![cfg(feature = "cuda")]

//! 生成管线：引擎 + tokenizer + 采样 + EOS/-n 语义 + 流式回调（run/serve 共用）。

use reinfer_core::DeviceId;
use reinfer_kernels::{CpuSamplerChain, RngState, SamplerChain, SamplerParams, select_sampler};
use reinfer_tokenizer::Tokenizer;

/// 由已采样 logits 计算 TokenOut（log-softmax 归一；top 默认 5）。
/// pub(crate)：S2-D 调度器循环（sched_loop）复用。
pub(crate) fn token_out(logits: &[f32], token: u32, top_n: usize) -> TokenOut {
    let mut maxv = f32::NEG_INFINITY;
    for &v in logits {
        if v > maxv {
            maxv = v;
        }
    }
    let mut z = 0.0f32;
    for &v in logits {
        z += (v - maxv).exp();
    }
    let lse = maxv + z.ln();
    let logp = |i: usize| -> f32 { logits[i] - lse };
    let mut top: Vec<(u32, f32)> = (0..logits.len()).map(|i| (i as u32, logp(i))).collect();
    let k = logits.len().min(top_n);
    if k >= 2 {
        top.select_nth_unstable_by(k - 1, |a, b| b.1.total_cmp(&a.1));
    }
    top.truncate(k);
    top.sort_by(|a, b| b.1.total_cmp(&a.1));
    TokenOut { token, logprob: logp(token as usize), top }
}

/// 采样参数（CLI 与 OpenAI 请求体统一映射面）。
#[derive(Clone, Debug)]
pub struct GenParams {
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub repeat_penalty: Option<f32>,
    /// D5 chain: OpenAI `frequency_penalty` (S3-2 serve surface).
    pub frequency_penalty: Option<f32>,
    /// D5 chain: OpenAI `presence_penalty` (S3-2 serve surface).
    pub presence_penalty: Option<f32>,
    pub seed: Option<u64>,
}

impl Default for GenParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: None,
            top_p: None,
            repeat_penalty: None,
            frequency_penalty: None,
            presence_penalty: None,
            seed: None,
        }
    }
}

/// 单 token 的输出详情（logprobs 面；经 log-softmax 归一）。
#[derive(Clone, Debug)]
pub struct TokenOut {
    /// sampled token id。
    pub token: u32,
    /// log p(token)（log-softmax）。
    pub logprob: f32,
    /// top-k (id, logprob)（采样时刻排序，不含 token 自身也可——含自身）。
    pub top: Vec<(u32, f32)>,
}

/// 生成统计。
#[derive(Clone, Debug)]
pub struct GenStat {
    pub tokens: usize,
    pub elapsed: std::time::Duration,
    pub first_token: Option<std::time::Duration>,
    pub stopped_by_eos: bool,
    /// S3-1: a `stop` sequence matched (the sampled token behind it is
    /// consumed but not emitted — OpenAI stop semantics).
    pub stopped_by_stop: bool,
}

/// 006-2 T3E: build the sampler chain for one generation. The GPU provider
/// (single-launch kernel) is preferred, and the CPU adapter is registered as
/// the automatic fallback — `select_sampler` orders by variant (GPU < CPU)
/// and composes a `FallbackSamplerChain` that re-dispatches
/// `SampleError::NotSupported` to the CPU path and counts `eager_fallback`.
/// GPU construction failure (no CUDA context / no nvcc / JIT error) degrades
/// to the CPU-only chain — never a silent wrong path.
fn build_sampler_chain(
    engine: &reinfer_cuda::engine::Engine,
    params: &SamplerParams,
) -> Result<Box<dyn SamplerChain>, String> {
    let mut providers: Vec<Box<dyn SamplerChain>> = Vec::new();
    let gpu = reinfer_cuda::arch::resolve_arch().ok().and_then(|arch| {
        reinfer_cuda::GpuSamplerChain::new(
            DeviceId::new(engine.device()),
            &arch,
            Some(std::env::temp_dir().join("reinfer-jit-sampler")),
        )
        .ok()
    });
    match gpu {
        Some(g) => providers.push(Box::new(g)),
        None => {
            eprintln!("reinfer: sampler: GPU chain unavailable — CPU chain only (006-2 T3E)");
        }
    }
    let cpu = CpuSamplerChain::new(params).map_err(|e| format!("sampler: CPU chain init: {e}"))?;
    providers.push(Box::new(cpu));
    select_sampler(providers).map_err(|e| format!("sampler: chain select: {e}"))
}

/// 流式生成：prefill prompt → 采样循环 → `sink(&str)` 增量文本回调。
/// `sink` 返回 false 即终止（管道关闭/客户端断流）。EOS/-n 恒定生效。
#[allow(clippy::too_many_arguments)]
pub fn generate_stream(
    engine: &mut reinfer_cuda::engine::Engine,
    tokenizer: &Tokenizer,
    prompt_ids: &[u32],
    params: &GenParams,
    eos_id: Option<u32>,
    // S3-1: OpenAI `stop` sequences (token ids; suffix match). Empty =
    // legacy EOS-only behavior (bit-identical).
    stop: &[Vec<u32>],
    max_tokens: u32,
    mut sink: impl FnMut(&str, Option<&TokenOut>) -> bool,
    logprobs_top_n: usize,
) -> Result<GenStat, String> {
    if prompt_ids.is_empty() {
        return Err("empty prompt".into());
    }
    // prefill：prompt 全部位置写 KV —— 006 T1: batched FMHA prefill first
    // (matching per-token semantics; diff gate: F16 cross-path & greedy 64/64).
    // Fallback keeps the per-token step path (dense/recoverable).
    match engine.prefill_batch(prompt_ids) {
        Ok(_) => {}
        Err(e) => {
            // per-token 重写同一 KV 位置（相同 token）幂等——安全回退
            eprintln!("prefill: batched path unavailable ({e}), falling back to per-token step");
            for (i, &tok) in prompt_ids.iter().enumerate() {
                engine.step(tok, i, i + 1).map_err(|e| format!("prefill: {e}"))?;
            }
        }
    }

    // 006-2 T3E: sampler chain — GPU preferred (single-launch kernel on the
    // engine's device), CPU adapter as automatic fallback; built once per
    // generation, outside the decode loop. A fresh chain per call mirrors the
    // previous construct-once sampler semantics (per-request penalty window).
    let sampler_params = SamplerParams {
        temperature: params.temperature,
        top_k: params.top_k,
        top_p: params.top_p,
        repeat_penalty: params.repeat_penalty,
        frequency_penalty: params.frequency_penalty,
        presence_penalty: params.presence_penalty,
        repeat_last_n: 64, // legacy pipeline penalty window (unchanged)
        seed: params.seed,
        ..SamplerParams::default()
    };
    let mut chain = build_sampler_chain(engine, &sampler_params)?;
    let mut rng = RngState::new(params.seed.unwrap_or(0));

    let mut generated: Vec<u32> = Vec::new();
    let mut cur = *prompt_ids.last().unwrap();
    // 014 S0-3b: the first decode step re-runs the last prompt position —
    // the idempotent rewrite of slot S-1 (KV cutoff = S) matches the
    // llama.cpp referee position semantics. The old pos = S start duplicated
    // the last prompt token (slots S-1 and S both held it — an off-by-one vs
    // llama.cpp, see bench/notes.md "duplicated last-prompt token").
    let mut pos = prompt_ids.len() - 1;
    let mut last_len = 0usize;
    let t0 = std::time::Instant::now();
    let mut first_token = None;
    let mut stopped_by_eos = false;
    let mut stopped_by_stop = false;
    while generated.len() < max_tokens as usize {
        let logits = engine.step(cur, pos, pos + 1).map_err(|e| format!("generate: {e}"))?;
        if first_token.is_none() {
            first_token = Some(t0.elapsed());
        }
        if logits.iter().all(|l| l.is_nan()) {
            return Err("logits contain only NaN — refuse to sample".into());
        }
        // Sample via the chain. The chain self-manages the penalty window
        // (the CPU adapter self-feeds, the GPU chain keeps its own history) —
        // the legacy post-sample feed() call is gone, no double feed.
        let view = engine.logits_view();
        let next = chain
            .sample(&view, &sampler_params, &mut rng)
            .map_err(|e| format!("sampler: {e}"))?
            .token;
        if Some(next) == eos_id {
            stopped_by_eos = true;
            break;
        }
        generated.push(next);
        // S3-1 stop: a matching suffix ends the generation — the sampled
        // token behind the match is consumed but not emitted.
        if let Some(pat) = stop.iter().find(|pat| generated.ends_with(pat.as_slice())) {
            generated.truncate(generated.len() - pat.len());
            stopped_by_stop = true;
            break;
        }
        cur = next;
        pos += 1;
        let full = tokenizer.decode_all(&generated);
        let delta = if full.len() > last_len {
            let d = full[last_len..].to_string();
            last_len = full.len();
            d
        } else {
            String::new()
        };
        let tokout =
            if logprobs_top_n > 0 { Some(token_out(&logits, next, logprobs_top_n)) } else { None };
        if !sink(&delta, tokout.as_ref()) {
            break; // 下游关闭（每次采样必发帧——delta 可为空）
        }
    }
    // 首 token 计入后（若 prefill 后即出）再补统计
    if generated.is_empty() {
        first_token = None;
    }
    Ok(GenStat {
        tokens: generated.len(),
        elapsed: t0.elapsed(),
        first_token,
        stopped_by_eos,
        stopped_by_stop,
    })
}

/// 渲染模型 chat 模板（tokenizer_config.json 的 `chat_template`；minijinja）。
/// 返回 Ok(None) = 无模板；Err = 模板本身不可用（调用方自行回退）。
#[cfg(feature = "cuda")] // 与 run 一致地仅在 cuda 后端路径使用
pub fn render_chat_template(
    dir: &std::path::Path,
    messages: &[serde_json::Value],
) -> Result<Option<String>, String> {
    let tcfg: serde_json::Value = std::fs::read(dir.join("tokenizer_config.json"))
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or(serde_json::Value::Null);
    let template =
        tcfg.get("chat_template").and_then(|v| v.as_str()).ok_or("no chat_template")?.to_string();
    let mut env = minijinja::Environment::new();
    env.set_undefined_behavior(minijinja::UndefinedBehavior::Lenient);
    // minijinja 无 Python 字符串方法 startswith/endswith（Qwen3 模板依赖）；
    // 重写为过滤器调用（`x.startswith(y)` → `x | startswith(y)`）并注册等价过滤器。
    let template =
        template.replace(".startswith(", " | startswith(").replace(".endswith(", " | endswith(");
    env.add_filter("startswith", |s: String, p: String| s.starts_with(&p));
    env.add_filter("endswith", |s: String, p: String| s.ends_with(&p));
    let tmpl = env.template_from_str(&template).map_err(|e| format!("jinja parse: {e}"))?;
    let rendered = tmpl
        .render(minijinja::value::Value::from_serialize(&serde_json::json!({
            "messages": messages,
            // vLLM chat-completions 默认 add_generation_prompt=true：追加
            // assistant 前缀（Qwen3 模板据 `add_generation_prompt` 渲染
            // `<|im_start|>assistant\n<think>…`；旧键 `generation` 被模板
            // 忽略 → 无前缀 → 模型无 EOS 靶标，014 D8）。
            "add_generation_prompt": true,
            "enable_thinking": false // 与 vLLM 侧 chat_template_kwargs 同构
        })))
        .map_err(|e| format!("jinja render: {e}"))?;
    Ok(Some(rendered))
}
