//! HF `config.json` → `LlamaConfig` 真模型验证（env-gated）。
//!
//! 运行：`REINFER_MODEL_DIR=/home/dora/.reinfer/models/Qwen/Qwen3-0.6B cargo test
//! -p reinfer-arch --test hf_config -- --ignored`
//!
//! 断言面：Qwen3-0.6B 官方 config 的全字段映射（hidden 1024/28 层/16:8 头/
//! head_dim 128/vocab 151936/rope 1e6/eps 1e-6/head_norm）；零模型名依赖。

#![allow(clippy::unwrap_used)] // 测试断言崩溃即失败

use reinfer_arch::llama::{from_hf_config, Architecture, RopeType};

fn load() -> serde_json::Value {
    let dir = std::env::var("REINFER_MODEL_DIR").expect("REINFER_MODEL_DIR env-gated");
    serde_json::from_slice(&std::fs::read(format!("{dir}/config.json")).unwrap())
        .expect("config.json parse")
}

#[test]
#[ignore = "hf smoke: env-gated real model"]
fn qwen3_config_maps() {
    let cfg = from_hf_config(&load()).expect("from_hf_config");
    assert_eq!(cfg.architecture, Architecture::Qwen3);
    assert_eq!(cfg.ctx_len, 40_960);
    assert_eq!(cfg.n_layer, 28);
    assert_eq!(cfg.hidden_size, 1_024);
    assert_eq!(cfg.q_heads, 16);
    assert_eq!(cfg.kv_heads, 8);
    assert_eq!(cfg.head_dim, 128);
    assert_eq!(cfg.value_dim, 128);
    assert_eq!(cfg.rope_dim, 128);
    assert_eq!(cfg.rope_theta, 1_000_000.0);
    assert_eq!(cfg.rope_type, RopeType::Neox);
    assert_eq!(cfg.rms_eps, 1e-6);
    assert_eq!(cfg.ffn_hidden, 3_072);
    assert_eq!(cfg.vocab_size, 151_936);
    assert_eq!(cfg.bos_id, Some(151_643));
    assert_eq!(cfg.eos_id, Some(151_645));
    assert!(cfg.head_norm, "Qwen3 必有 q/k head norm");
    assert!(cfg.unk_id.is_none());
}

#[test]
#[ignore = "hf smoke: env-gated real model"]
fn qwen3_unknown_arch_rejected() {
    let mut v = load();
    v["model_type"] = serde_json::json!("gemma3");
    v["architectures"] = serde_json::json!(["Gemma3ForCausalLM"]);
    let err = from_hf_config(&v).unwrap_err();
    assert!(err.to_string().contains("unknown"), "{}", err);
}
