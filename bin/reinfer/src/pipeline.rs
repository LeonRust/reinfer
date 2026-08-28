//! 生成管线：引擎 + tokenizer + 采样 + EOS/-n 语义 + 流式回调（run/serve 共用）。
//!
//! 依赖 reinfer-cuda（Engine/argmax）——整个模块按 `cuda` feature 门控。

#![cfg(feature = "cuda")]

//! 生成管线：引擎 + tokenizer + 采样 + EOS/-n 语义 + 流式回调（run/serve 共用）。

use reinfer_tokenizer::Tokenizer;

/// 采样参数（CLI 与 OpenAI 请求体统一映射面）。
#[derive(Clone, Debug)]
pub struct GenParams {
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub repeat_penalty: Option<f32>,
    pub seed: Option<u64>,
}

impl Default for GenParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: None,
            top_p: None,
            repeat_penalty: None,
            seed: None,
        }
    }
}

/// 生成统计。
#[derive(Clone, Debug)]
pub struct GenStat {
    pub tokens: usize,
    pub elapsed: std::time::Duration,
    pub first_token: Option<std::time::Duration>,
    pub stopped_by_eos: bool,
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
    max_tokens: u32,
    mut sink: impl FnMut(&str) -> bool,
) -> Result<GenStat, String> {
    if prompt_ids.is_empty() {
        return Err("empty prompt".into());
    }
    // prefill：prompt 全部位置写 KV
    for (i, &tok) in prompt_ids.iter().enumerate() {
        engine
            .step(tok, i, i + 1)
            .map_err(|e| format!("prefill: {e}"))?;
    }

    let mut sampler = if params.temperature > 0.0 {
        Some(
            reinfer_samplers::Sampler::new(&reinfer_samplers::SamplingParams {
                temperature: params.temperature,
                top_k: params.top_k,
                top_p: params.top_p,
                repeat_penalty: params.repeat_penalty,
                repeat_last_n: 64,
                seed: params.seed,
            })
            .map_err(|e| format!("sampler init: {e}"))?,
        )
    } else {
        None
    };

    let mut generated: Vec<u32> = Vec::new();
    let mut cur = *prompt_ids.last().unwrap();
    let mut pos = prompt_ids.len();
    let mut last_len = 0usize;
    let t0 = std::time::Instant::now();
    let mut first_token = None;
    let mut stopped_by_eos = false;
    while generated.len() < max_tokens as usize {
        let logits = engine
            .step(cur, pos, pos + 1)
            .map_err(|e| format!("generate: {e}"))?;
        if first_token.is_none() {
            first_token = Some(t0.elapsed());
        }
        if logits.iter().all(|l| l.is_nan()) {
            return Err("logits contain only NaN — refuse to sample".into());
        }
        // 采样（greedy 与 argmax-first tie-break 语义一致）
        let next = match &mut sampler {
            Some(s) => {
                let t = s.sample(&logits).map_err(|e| format!("sampler: {e}"))?;
                s.feed(t);
                t
            }
            None => reinfer_cuda::engine::argmax_first(&logits),
        };
        if Some(next) == eos_id {
            stopped_by_eos = true;
            break;
        }
        generated.push(next);
        cur = next;
        pos += 1;
        let full = tokenizer.decode_all(&generated);
        if full.len() > last_len {
            let delta = full[last_len..].to_string();
            last_len = full.len();
            if !sink(&delta) {
                break; // 下游关闭
            }
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
    let template = tcfg
        .get("chat_template")
        .and_then(|v| v.as_str())
        .ok_or("no chat_template")?
        .to_string();
    let mut env = minijinja::Environment::new();
    env.set_undefined_behavior(minijinja::UndefinedBehavior::Lenient);
    let tmpl = env
        .template_from_str(&template)
        .map_err(|e| format!("jinja parse: {e}"))?;
    let rendered = tmpl
        .render(minijinja::value::Value::from_serialize(&serde_json::json!({
            "messages": messages,
            "generation": false
        })))
        .map_err(|e| format!("jinja render: {e}"))?;
    Ok(Some(rendered))
}
