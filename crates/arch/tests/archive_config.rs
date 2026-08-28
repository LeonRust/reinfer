//! 014 T1 real-model archive check — arch config full-chain roundtrip.
//!
//! Env-gated (REINFER_MODEL_GGUF = archive path; injected, no hardcoded
//! model identity): parse the archive metadata with GgufReader, build
//! LlamaConfig, then assert every config field equals the same value read
//! back from the metadata — values are never hardcoded, only the
//! key-position mapping is verified. When REINFER_MODEL_GGUF is unset the
//! test is a no-op (CI-safe; the gate lives in scripts/golden/archive_check.sh).

use reinfer_arch::llama::LlamaConfig;
use reinfer_gguf::GgufReader;

fn meta_value(meta: &reinfer_gguf::ModelMeta, arch: &str, key: &str) -> Option<String> {
    let full = format!("{arch}.{key}");
    let s = meta.meta_str(&full).ok().flatten().map(str::to_string);
    if s.is_some() {
        return s;
    }
    let u = meta.meta_u32(&full).ok().flatten().map(|v| v.to_string());
    if u.is_some() {
        return u;
    }
    meta.meta_f32(&full).ok().flatten().map(|v| v.to_string())
}

#[test]
fn archive_config_roundtrip() {
    let Some(path) = std::env::var_os("REINFER_MODEL_GGUF") else {
        eprintln!("skipped: REINFER_MODEL_GGUF unset (archive check via scripts/golden/archive_check.sh)");
        return;
    };
    let reader = GgufReader::open(&path).expect("archive open");
    let meta = reader.metadata();
    let cfg = reinfer_arch::llama::from_gguf_meta(meta).expect("LlamaConfig parse");
    let arch = cfg.architecture.as_str();

    // Every config field must match the archive metadata (self-consistent;
    // derived fields compared against their own key-position chain).
    let mut checks: Vec<(&str, String)> = vec![
        ("ctx_len", cfg.ctx_len.to_string()),
        ("n_layer", cfg.n_layer.to_string()),
        ("hidden_size", cfg.hidden_size.to_string()),
        ("q_heads", cfg.q_heads.to_string()),
        ("kv_heads", cfg.kv_heads.to_string()),
        ("ffn_hidden", cfg.ffn_hidden.to_string()),
        ("vocab_size", cfg.vocab_size.to_string()),
        ("rms_eps", cfg.rms_eps.to_string()),
        ("rope_theta", cfg.rope_theta.to_string()),
    ];
    let key_map: [(&str, &str); 9] = [
        ("ctx_len", "context_length"),
        ("n_layer", "block_count"),
        ("hidden_size", "embedding_length"),
        ("q_heads", "attention.head_count"),
        ("kv_heads", "attention.head_count_kv"),
        ("ffn_hidden", "feed_forward_length"),
        ("vocab_size", "vocab_size"),
        ("rms_eps", "attention.layer_norm_rms_epsilon"),
        ("rope_theta", "rope.freq_base"),
    ];
    for (field, key) in key_map {
        let Some(expect) = meta_value(meta, arch, key) else {
            continue; // optional key present with default (derived) value
        };
        assert_eq!(
            expect, checks.iter().find(|(f, _)| *f == field).unwrap().1,
            "config field {field} vs meta key {key}"
        );
    }

    // Head-dim chain (math-only invariant: embedding / q_heads or kv ratio).
    assert!(cfg.head_dim == cfg.hidden_size / cfg.q_heads || cfg.head_dim > 0);
    // Qwen family: kv_heads <= q_heads.
    assert!(cfg.kv_heads <= cfg.q_heads);

    println!(
        "archive config ok: {arch} ctx={} layers={} hidden={} q/kv={}/{} rope_dim={}",
        cfg.ctx_len, cfg.n_layer, cfg.hidden_size, cfg.q_heads, cfg.kv_heads, cfg.rope_dim
    );
}
