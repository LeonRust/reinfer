//! 014 L3：Qwen3-0.6B（safetensors）CUDA 真机冒烟。
//!
//! 运行：`CUDA_VISIBLE_DEVICES=0 REINFER_MODEL_DIR=/home/dora/.reinfer/models/Qwen/Qwen3-0.6B
//! cargo test -p reinfer-cuda --features cuda --test engine_smoke -- --ignored --test-threads=1`
//!
//! 判据面（可执行性优先——首版精度档为记录档，见 014 notes r2）：
//! ① 装载（config→LlamaConfig + 权重 f16 化上传）成功；
//! ② prefill 首步 logits 有限（无 NaN 传播——真模型应全部有限）；
//! ③ argmax < vocab_size；tokenizer 域全部落在 embedding 行 内；
//! ④ 生成 n=16 后按 EOS/-n 终止；解码串可打印（非空）；
//! ⑤ 两步复跑逐 token 确定（双引擎同设备同输入 — decode 确定性档）。
//! ⑥ 性能数据打印（token/s —— 记录档，无 gate）。

#![cfg(feature = "cuda")]
#![allow(clippy::unwrap_used)] // 测试断言崩溃即失败
#![allow(clippy::print_stdout)] // 冒烟输出

mod gpu {
    use reinfer_core::DeviceId;
    use reinfer_cuda::engine::{argmax_first, Engine, EngineError};
    use reinfer_cuda::{CudaContext, CudaStream};
    use reinfer_tokenizer::Tokenizer;
    use std::time::Instant;

    fn load_all(model_dir: &std::path::Path) -> (Engine, Tokenizer, serde_json::Value) {
        let ctx = CudaContext::init(DeviceId::new(0)).expect("ctx");
        let dev = ctx.device_id();
        let stream = CudaStream::new(dev).expect("stream");
        let _ = stream.synchronize().expect("sync");

        let tok: serde_json::Value = serde_json::from_slice(
            &std::fs::read(model_dir.join("tokenizer.json")).expect("tokenizer.json"),
        )
        .expect("tokenizer json");
        let cfg: serde_json::Value = serde_json::from_slice(
            &std::fs::read(model_dir.join("tokenizer_config.json")).expect("tokenizer_config.json"),
        )
        .expect("tokcfg json");
        let tokenizer = Tokenizer::from_hf_json(&tok, &cfg).expect("hf tokenizer");

        let engine = Engine::load(
            dev,
            &reinfer_cuda::arch::resolve_arch().expect("arch"),
            Some(std::env::temp_dir().join("reinfer-jit-dense")),
            model_dir,
            4096,
        )
        .expect("engine load");
        (engine, tokenizer, cfg)
    }

    fn model_dir() -> std::path::PathBuf {
        std::path::PathBuf::from(std::env::var("REINFER_MODEL_DIR").expect("REINFER_MODEL_DIR"))
    }

    #[test]
    #[ignore = "gpu.yml: l3-smoke / qwen3-0.6b"]
    fn engine_generates_sane_text() {
        let (mut engine, tokenizer, _cfg) = load_all(&model_dir());
        let prompt = "Hello";
        let ids = tokenizer.encode(prompt, false).expect("encode");
        assert!(!ids.is_empty());

        // ① prefill 首步 logits 有限
        let logits = engine.step(ids[0], 0, 1).expect("prefill step");
        assert!(logits.iter().all(|l| l.is_finite()), "logits must be finite");
        assert!(logits.iter().all(|l| !l.is_nan()), "no NaN");
        let argmax = argmax_first(&logits);
        assert!((argmax as usize) < engine.config().vocab_size);

        // ④ 生成（prompt 已预填一步——完整 generate 独立重跑）
        let eos = engine.config().eos_id.or_else(|| tokenizer.eos_token());
        let t0 = Instant::now();
        let out = engine.generate(&ids, 16, eos, 0.0).expect("generate");
        let dt = t0.elapsed();
        assert!(!out.is_empty(), "generation should produce tokens");
        assert!(out.len() <= 16);
        for &t in &out {
            assert!((t as usize) < engine.config().vocab_size, "token OOV {t}");
        }
        let text = tokenizer.decode_all(&out);
        println!(
            "prompt {prompt:?} -> {} tokens ({text:?}) in {dt:?} (~{:.1} tok/s)",
            out.len(),
            out.len() as f64 / dt.as_secs_f64()
        );
        assert!(!text.is_empty(), "decoded text non-empty");
    }

    #[test]
    #[ignore = "gpu.yml: l3-smoke / determinism"]
    fn engine_deterministic_two_passes() {
        let (mut engine, tokenizer, _) = load_all(&model_dir());
        let ids = tokenizer.encode("The quick brown fox", false).expect("encode");
        let eos = engine.config().eos_id;
        let a = engine.generate(&ids, 8, eos, 0.0).expect("pass a");
        let b = engine.generate(&ids, 8, eos, 0.0).expect("pass b");
        assert_eq!(a, b, "two passes must be identical (determinism)");
        println!("deterministic: {a:?}");
    }

    #[test]
    #[ignore = "gpu.yml: l3-smoke / oov"]
    fn engine_rejects_oov_token() {
        let (mut engine, _, _) = load_all(&model_dir());
        let vocab = engine.config().vocab_size as u32;
        let err = engine.step(vocab, 0, 1).err().expect("oov rejected");
        assert!(matches!(err, EngineError::EmbeddingOov(_)));
    }

    #[test]
    #[ignore = "gpu.yml: l3-smoke / param"]
    fn engine_rejects_temperature() {
        let (mut engine, tokenizer, _) = load_all(&model_dir());
        let ids = tokenizer.encode("Hello", false).expect("encode");
        let err = engine.generate(&ids, 4, None, 0.7).err().expect("temp rejected");
        assert!(matches!(err, EngineError::Sts(_)));
    }
}
