//! JitGemm engine A/B — the decode step's m=1 GEMMs through the
//! `gemv_m1_f16f32` kernel vs the cublas path (S1-6).
//!
//! Gates:
//!   - determinism: two 16-token greedy passes with jgemm on must produce
//!     identical token sequences (bit-level determinism);
//!   - no fallback: jgemm launch failures == 0 during generation.
//!
//! Records (printed, not gated): the token/text sequence with jgemm on vs
//! `REINFER_JGEMM=off`. The reduction order differs from cublas, so a
//! ~1e-5-level logits drift can flip a near-tie argmax; the report
//! evaluates any divergence against the D7 criterion table (text sequences
//! expected to stay identical on 16 tokens).
//!
//! Run: CUDA_VISIBLE_DEVICES=0 REINFER_MODEL_DIR=/home/dora/.reinfer/models/
//! Qwen/Qwen3-0.6B cargo test -p reinfer-cuda --features cuda
//! --test jgemm_engine -- --ignored --test-threads=1 --nocapture

#![cfg(feature = "cuda")]
#![allow(clippy::unwrap_used)] // 测试断言崩溃即失败
#![allow(clippy::print_stdout)] // A/B 冒烟输出

mod gpu {
    use reinfer_core::DeviceId;
    use reinfer_cuda::engine::Engine;
    use reinfer_cuda::{CudaContext, CudaStream};
    use reinfer_tokenizer::Tokenizer;

    const JGEMM_ENV: &str = "REINFER_JGEMM";

    fn model_dir() -> std::path::PathBuf {
        std::path::PathBuf::from(std::env::var("REINFER_MODEL_DIR").expect("REINFER_MODEL_DIR"))
    }

    /// Load the engine with REINFER_JGEMM set to `jgemm_value` (read once
    /// at load; the process env is ours — tests must run single-threaded).
    fn load(jgemm_value: &str) -> Engine {
        // SAFETY (test-only): --test-threads=1 keeps the env mutation
        // single-threaded; the value is read once at Engine::load.
        unsafe { std::env::set_var(JGEMM_ENV, jgemm_value) };
        let ctx = CudaContext::init(DeviceId::new(0)).expect("ctx");
        let dev = ctx.device_id();
        let stream = CudaStream::new(dev).expect("stream");
        let _ = stream.synchronize().expect("sync");
        let dir = model_dir();
        Engine::load(
            dev,
            &reinfer_cuda::arch::resolve_arch().expect("arch"),
            Some(std::env::temp_dir().join("reinfer-jit-dense")),
            &dir,
            4096,
        )
        .expect("engine load")
    }

    fn tokenizer() -> Tokenizer {
        let dir = model_dir();
        let tok: serde_json::Value = serde_json::from_slice(
            &std::fs::read(dir.join("tokenizer.json")).expect("tokenizer.json"),
        )
        .expect("tokenizer json");
        let cfg: serde_json::Value = serde_json::from_slice(
            &std::fs::read(dir.join("tokenizer_config.json")).expect("tokenizer_config.json"),
        )
        .expect("tokcfg json");
        Tokenizer::from_hf_json(&tok, &cfg).expect("hf tokenizer")
    }

    /// 16-token greedy generation (temperature 0 -> argmax).
    fn gen16(engine: &mut Engine, ids: &[u32]) -> Vec<u32> {
        let eos = engine.config().eos_id;
        engine.generate(ids, 16, eos, 0.0).expect("generate")
    }

    #[test]
    #[ignore = "gpu.yml: l3-jgemm / engine-ab"]
    fn jgemm_text_determinism_and_ab() {
        let tok = tokenizer();
        let ids = tok.encode("Hello", false).expect("encode");
        assert!(!ids.is_empty());

        // A: jgemm on — two passes must be bit-identical (determinism gate).
        let mut eng_on = load("on");
        assert!(eng_on.jgemm_enabled(), "REINFER_JGEMM=on must load jgemm");
        let a = gen16(&mut eng_on, &ids);
        let a2 = gen16(&mut eng_on, &ids);
        assert_eq!(a, a2, "jgemm two passes must be identical (determinism)");
        let fb = eng_on.jgemm_fallbacks();
        assert_eq!(fb, 0, "jgemm launch failures during generation");
        println!(
            "jgemm on : {} tokens {:?} (text {:?}), fallbacks {fb}",
            a.len(),
            a,
            tok.decode_all(&a)
        );
        drop(eng_on);

        // B: REINFER_JGEMM=off — the original cublas path.
        let mut eng_off = load("off");
        assert!(!eng_off.jgemm_enabled(), "REINFER_JGEMM=off must not load jgemm");
        let b = gen16(&mut eng_off, &ids);
        println!("jgemm off: {} tokens {:?} (text {:?})", b.len(), b, tok.decode_all(&b));
        drop(eng_off);

        // Record: sequence equality across the two paths. An argmax flip
        // on a near-tie (logits drifting at ~1e-5 from the order change)
        // is evaluated against the D7 criterion table and reported; the
        // determinism gate above is the hard requirement.
        if a == b {
            println!("A/B verdict: 16-token greedy sequences IDENTICAL");
        } else {
            let common = a.iter().zip(&b).take_while(|(x, y)| x == y).count();
            println!(
                "A/B verdict: sequences DIFFER after {common} shared tokens (jgemm on vs off) \
                 — drift record (D7 evaluation in the report)"
            );
        }
    }
}
