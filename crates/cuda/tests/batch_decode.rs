//! S2-B: batch decode step acceptance tests (spec 005 S2-B — B requests x
//! 1 token merged forward).
//!
//! Run (RTX 5090 / sm_120a JIT env):
//! ```
//! REINFER_CUDA_NVCC=/usr/local/cuda-13.2/bin/nvcc \
//! REINFER_MODEL_DIR=/home/dora/.reinfer/models/Qwen/Qwen3-0.6B \
//! cargo test -p reinfer-cuda --features cuda --test batch_decode -- \
//!     --ignored --test-threads=1 --nocapture
//! ```
//!
//! Acceptance surface (S2-B+):
//! ① B=1 falls back to the single-request path — bit-level equality over
//!    16 tokens (same input, step by step);
//! ② B=4 numerics vs independent single-request runs — bit-level gate: the
//!    batch path's kernels are per-element identical to the single path's
//!    (jgemm m=B uses the same (n, k) -> nslabs decomposition and the same
//!    block arithmetic as the m=1 path; the flash/rope/kv kernels were
//!    verified element-identical), so every request must match its
//!    independent run exactly;
//! ③ B=4 determinism — a double run must be bitwise identical;
//! ④ B=4 per-step time vs 4 x B=1 per-step time, against both the default
//!    layer-fused single path and the split kernels the batch path mirrors
//!    (record tier, no gate). REINFER_BATCH_PROF=1 prints the per-segment
//!    qkv/attn/rest/lm breakdown for each batch step;
//! ⑤ B=20 extension — 20/20 bitwise vs single references, double-run
//!    determinism, and per-request time <= 2x the B=4 per-request time.

#![cfg(feature = "cuda")]
#![allow(clippy::unwrap_used)] // 测试断言崩溃即失败
#![allow(clippy::print_stdout)] // 记录档输出

mod gpu {
    use reinfer_core::DeviceId;
    use reinfer_cuda::buffer::MemRef;
    use reinfer_cuda::engine::{BatchReq, Engine, SegRef};
    use reinfer_cuda::{CudaContext, DeviceBuffer, copy};
    use reinfer_tokenizer::Tokenizer;
    use std::time::Instant;

    /// Decode-step KV page size — mirrors engine.rs `BLOCK_LEN` (32).
    const BLOCK_LEN: usize = 32;

    fn model_dir() -> std::path::PathBuf {
        std::path::PathBuf::from(std::env::var("REINFER_MODEL_DIR").expect("REINFER_MODEL_DIR"))
    }

    fn tokenizer() -> Tokenizer {
        let tok: serde_json::Value = serde_json::from_slice(
            &std::fs::read(model_dir().join("tokenizer.json")).expect("tokenizer.json"),
        )
        .expect("tokenizer json");
        let tcfg: serde_json::Value = serde_json::from_slice(
            &std::fs::read(model_dir().join("tokenizer_config.json"))
                .expect("tokenizer_config.json"),
        )
        .expect("tokcfg json");
        Tokenizer::from_hf_json(&tok, &tcfg).expect("hf tokenizer")
    }

    fn load(dev: u32, max_kv: usize) -> Engine {
        let arch = reinfer_cuda::arch::resolve_arch().expect("arch");
        Engine::load(
            DeviceId::new(dev),
            &arch,
            Some(std::env::temp_dir().join("reinfer-jit-batch")),
            &model_dir(),
            max_kv,
        )
        .expect("engine load")
    }

    /// Long-enough prompt for the 16-token (B=1), 18-token (B=4) and
    /// 34-token (B=20, 20 distinct request tokens) acceptance windows.
    const PROMPT: &str = "The quick brown fox jumps over the lazy dog near \
        the river while the autumn leaves drift slowly down and settle on \
        the quiet ground and the wind carries the first cold breath of \
        evening across the empty meadow";

    fn bitwise(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
    }

    fn max_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
    }

    /// ① B=1 fallback: bit-level equality with the single-request path over
    /// 16 tokens (the batch B=1 path must be exactly `Engine::step`).
    #[test]
    #[ignore = "gpu.yml: s2b-batch-decode / b1-bitwise"]
    fn batch_b1_bitwise_vs_step() {
        let _ctx = CudaContext::init(DeviceId::new(0)).expect("ctx");
        let tokenizer = tokenizer();
        let mut engine = load(0, 4096);
        let ids = tokenizer.encode(PROMPT, false).expect("encode");
        assert!(ids.len() >= 16, "prompt must encode to >= 16 tokens");
        let ids = &ids[..16];
        for (i, &t) in ids.iter().enumerate() {
            let single = engine.step(t, i, i + 1).expect("single step");
            let batch = engine
                .batch_decode_step(&[BatchReq { token: t, pos: i, kv: SegRef::engine(i + 1) }])
                .expect("batch B=1 step");
            assert_eq!(batch.len(), 1, "B=1 returns one logits row");
            assert!(
                bitwise(&single, &batch[0]),
                "pos {i}: B=1 batch diverged from the single path bitwise \
                 (max |diff| {})",
                max_diff(&single, &batch[0])
            );
        }
        println!("b1: 16/16 steps bitwise == single path");
    }

    /// Build an engine prefilled with `prefix` tokens and a batch pool of
    /// `n` per-request segments cloned from the engine's KV (bit-identical
    /// prefix). Returns (engine, segment buffers, prefix ids).
    fn prefill_and_seed(dev: u32, prefix: &[u32], n: usize) -> (Engine, Vec<DeviceBuffer>) {
        let mut engine = load(dev, 256); // pp = 8 pages/layer
        // Prefill the engine's own pool with the shared prefix.
        for (s, &t) in prefix.iter().enumerate() {
            engine.step(t, s, s + 1).expect("prefill step");
        }
        // Clone the engine's KV (K region + V region — one segment's
        // geometry: total_pages = n_layer*pp) into n per-request pools.
        let cfg = engine.config();
        let kv = engine.kv_store();
        let seg_bytes =
            cfg.n_layer * (256 / BLOCK_LEN) * BLOCK_LEN * cfg.kv_heads * cfg.head_dim * 4; // K+V, f16
        assert_eq!(seg_bytes, kv.data.size(), "segment geometry == engine pool");
        let segs: Vec<DeviceBuffer> = (0..n)
            .map(|_| DeviceBuffer::alloc(DeviceId::new(dev), seg_bytes).expect("seg alloc"))
            .collect();
        for seg in &segs {
            // D2D whole-buffer copy (legacy default stream orders after the
            // engine's blocking stream; the batch step below orders after
            // this copy).
            copy(&mut MemRef::Device(seg), &MemRef::Device(&kv.data), seg_bytes, None)
                .expect("segment seed copy");
        }
        (engine, segs)
    }

    /// ② B=4 numerics: each request vs its independent single-request run
    /// (same prefix, same KV — the batch segments are bit-clones of the
    /// engine pool). S2-B+ gate: bit-level — all 4 requests must match
    /// their single-path references exactly (the S2-B record of m=B cublas
    /// D7 drift amplified to ~1e-2 logit scale is gone: jgemm m=B is
    /// per-element the m=1 arithmetic).
    #[test]
    #[ignore = "gpu.yml: s2b-batch-decode / b4-vs-single"]
    fn batch_b4_vs_single_reference() {
        let dev = 0;
        let _ctx = CudaContext::init(DeviceId::new(dev)).expect("ctx");
        let prefix: Vec<u32> = tokenizer().encode(PROMPT, false).expect("encode");
        assert!(prefix.len() >= 18, "prompt must encode to >= 18 tokens");
        let prefix = &prefix[..14];
        let toks = [prefix[0], prefix[1], prefix[2], prefix[3]];
        let (mut engine, segs) = prefill_and_seed(dev, prefix, toks.len());

        // Independent references (single-request path, its own pool; each
        // reference attends over the 14-token prefix + its own write).
        let refs: Vec<Vec<f32>> =
            toks.iter().map(|&t| engine.step(t, 14, 15).expect("reference step")).collect();

        // The batch call: same 4 requests on the cloned segments.
        let reqs: Vec<BatchReq> = segs
            .iter()
            .enumerate()
            .map(|(b, seg)| BatchReq {
                token: toks[b],
                pos: 14,
                kv: SegRef { kv: seg.as_ptr() as *mut u16, base_pages: 0, len: 15 },
            })
            .collect();
        let out = engine.batch_decode_step(&reqs).expect("batch step");
        assert_eq!(out.len(), toks.len(), "B=4 returns 4 logits rows");

        let mut bit_exact = 0;
        let mut worst = 0.0f32;
        for b in 0..toks.len() {
            let d = max_diff(&refs[b], &out[b]);
            worst = worst.max(d);
            // S2-B+ bit-level gate: every batch kernel (gather_rows,
            // rms_norm_rows, cast_split_qkv, rope_batch, kv_write_batch,
            // flash_batch, gemv_mb) is per-element the single path's
            // arithmetic, and jgemm m=B derives (ncols, nslabs) from (n, k)
            // exactly as the m=1 launch does — so the same (n, k) per
            // projection reproduces the single path bit-for-bit.
            assert!(
                bitwise(&refs[b], &out[b]),
                "req {b}: batch diverged from the single path bitwise \
                 (max |diff| {d:e})"
            );
            bit_exact += 1;
        }
        assert_eq!(bit_exact, toks.len(), "not all requests bitwise == single");
        println!(
            "b4: {bit_exact}/{} requests bitwise == independent single runs; \
             worst max|diff| {worst:e}",
            toks.len()
        );
    }

    /// ③ B=4 determinism: a double run with identical input must be bitwise
    /// identical (both runs write the same kv for the same slot — the
    /// second run sees the first run's writes, which are identical).
    #[test]
    #[ignore = "gpu.yml: s2b-batch-decode / b4-determinism"]
    fn batch_b4_determinism() {
        let dev = 0;
        let _ctx = CudaContext::init(DeviceId::new(dev)).expect("ctx");
        let prefix: Vec<u32> = tokenizer().encode(PROMPT, false).expect("encode");
        assert!(prefix.len() >= 18, "prompt must encode to >= 18 tokens");
        let prefix = &prefix[..14];
        let toks = [prefix[0], prefix[1], prefix[2], prefix[3]];
        let (mut engine, segs) = prefill_and_seed(dev, prefix, toks.len());
        let reqs: Vec<BatchReq> = segs
            .iter()
            .enumerate()
            .map(|(b, seg)| BatchReq {
                token: toks[b],
                pos: 14,
                kv: SegRef { kv: seg.as_ptr() as *mut u16, base_pages: 0, len: 15 },
            })
            .collect();
        let a = engine.batch_decode_step(&reqs).expect("run a");
        let b = engine.batch_decode_step(&reqs).expect("run b");
        assert_eq!(a.len(), b.len());
        for i in 0..a.len() {
            assert!(
                bitwise(&a[i], &b[i]),
                "req {i}: double run diverged bitwise (max |diff| {})",
                max_diff(&a[i], &b[i])
            );
        }
        println!("b4: double run bitwise identical");
    }

    /// ④ B=4 per-step time vs 4 x B=1 per-step time (record tier). The
    /// projections run batched m=B jgemm and the attention runs the batch
    /// flash kernel (grid +B dimension, per-request page tables/positions).
    /// Two single-path references: the default layer-fused kernel (1 kernel
    /// per layer — the current `step`) and the split kernels (the same
    /// kernel structure the batch path mirrors; the GEMM-batching win is
    /// measured against this one). Per-segment breakdown: run with
    /// REINFER_BATCH_PROF=1 to print qkv/attn/rest/lm per batch step.
    #[test]
    #[ignore = "gpu.yml: s2b-batch-decode / b4-perf-micro"]
    fn batch_b4_perf_micro() {
        let dev = 0;
        let _ctx = CudaContext::init(DeviceId::new(dev)).expect("ctx");
        let prefix: Vec<u32> = tokenizer().encode(PROMPT, false).expect("encode");
        assert!(prefix.len() >= 18, "prompt must encode to >= 18 tokens");
        let prefix = &prefix[..14];
        let toks = [prefix[0], prefix[1], prefix[2], prefix[3]];
        let (mut engine, segs) = prefill_and_seed(dev, prefix, toks.len());

        // Split single-path engine (same kernel structure as the batch
        // path; env vars read at load, restored afterwards for the other
        // tests' runs — single-threaded test process).
        let mut engine_split = {
            unsafe {
                std::env::set_var("REINFER_LAYER_FUSED", "off");
                std::env::set_var("REINFER_FUSED", "off");
            }
            let mut e = load(dev, 256);
            for (s, &t) in prefix.iter().enumerate() {
                e.step(t, s, s + 1).expect("prefill split");
            }
            e
        };
        unsafe {
            std::env::set_var("REINFER_LAYER_FUSED", "on");
            std::env::set_var("REINFER_FUSED", "on");
        }

        const N: usize = 50;
        let reqs_at = |s: usize| -> Vec<BatchReq> {
            segs.iter()
                .enumerate()
                .map(|(b, seg)| BatchReq {
                    token: toks[b],
                    pos: 14 + s,
                    kv: SegRef { kv: seg.as_ptr() as *mut u16, base_pages: 0, len: 15 + s },
                })
                .collect()
        };

        // Warmup (JIT/cublas warm).
        for s in 0..5 {
            engine.batch_decode_step(&reqs_at(s)).expect("batch warmup");
        }
        let t0 = Instant::now();
        for s in 0..N {
            engine.batch_decode_step(&reqs_at(s)).expect("batch step");
        }
        let dt_batch = t0.elapsed();

        // 4 x N single-request steps (same positions/kv lens; the shared
        // engine pool interleaves the 4 sequences — values differ from the
        // batch run, the GEMM shapes and attention windows are identical).
        for b in 0..toks.len() {
            for s in 0..5 {
                engine.step(toks[b], 14 + s, 15 + s).expect("single warmup");
            }
        }
        let t1 = Instant::now();
        for b in 0..toks.len() {
            for s in 0..N {
                engine.step(toks[b], 14 + s, 15 + s).expect("single step");
            }
        }
        let dt_single_fused = t1.elapsed();

        for b in 0..toks.len() {
            for s in 0..5 {
                engine_split.step(toks[b], 14 + s, 15 + s).expect("split warmup");
            }
        }
        let t2 = Instant::now();
        for b in 0..toks.len() {
            for s in 0..N {
                engine_split.step(toks[b], 14 + s, 15 + s).expect("split step");
            }
        }
        let dt_single_split = t2.elapsed();

        let per_batch = dt_batch.as_secs_f64() * 1000.0 / N as f64;
        let per_fused_x4 = dt_single_fused.as_secs_f64() * 1000.0 / (4 * N) as f64;
        let per_split_x4 = dt_single_split.as_secs_f64() * 1000.0 / (4 * N) as f64;
        println!(
            "b4 perf micro: {N} steps x B=4 in {dt_batch:?} ({per_batch:.2} ms/step)\n\
             \tvs 4x{N} steps x B=1 layer-fused (default step) in {dt_single_fused:?} \
             ({per_fused_x4:.2} ms/4 tok) -> {:.2}x\n\
             \tvs 4x{N} steps x B=1 split kernels in {dt_single_split:?} \
             ({per_split_x4:.2} ms/4 tok) -> {:.2}x (GEMM-batching reference)",
            per_fused_x4 / per_batch,
            per_split_x4 / per_batch
        );
    }

    /// ⑤ B=20 extension (S2-B+ acceptance: per-request time <= 2x B=4, with
    /// the same numerics guarantees as B=4). 20 requests take 20 distinct
    /// prompt tokens, each attending its own cloned segment:
    /// - 20/20 bitwise vs the independent single-request runs;
    /// - double run bitwise identical (determinism);
    /// - per-request ms/step at B=20 <= 2x the B=4 per-request ms/step
    ///   (measured at ~0.5x: the GEMM cost scales with B, but the lm_head
    ///   W slice is read once per cell and shared across rows in L2).
    #[test]
    #[ignore = "gpu.yml: s2b-batch-decode / b20-extension"]
    fn batch_b20_extension() {
        let dev = 0;
        let _ctx = CudaContext::init(DeviceId::new(dev)).expect("ctx");
        let ids: Vec<u32> = tokenizer().encode(PROMPT, false).expect("encode");
        assert!(ids.len() >= 34, "prompt must encode to >= 34 tokens");
        let prefix = &ids[..14];
        let toks = ids[14..34].to_vec(); // 20 distinct tokens
        let (mut engine, segs) = prefill_and_seed(dev, prefix, toks.len());
        assert_eq!(segs.len(), 20);

        // Independent references (single-request path, its own pool).
        let refs: Vec<Vec<f32>> =
            toks.iter().map(|&t| engine.step(t, 14, 15).expect("reference step")).collect();

        let reqs_at = |s: usize, segs: &[DeviceBuffer]| -> Vec<BatchReq> {
            segs.iter()
                .enumerate()
                .map(|(b, seg)| BatchReq {
                    token: toks[b],
                    pos: 14 + s,
                    kv: SegRef { kv: seg.as_ptr() as *mut u16, base_pages: 0, len: 15 + s },
                })
                .collect()
        };

        // Bitwise vs the single references (pos 14, len 15).
        let out = engine.batch_decode_step(&reqs_at(0, &segs)).expect("batch step");
        assert_eq!(out.len(), toks.len(), "B=20 returns 20 logits rows");
        for b in 0..toks.len() {
            assert!(
                bitwise(&refs[b], &out[b]),
                "req {b}: B=20 batch diverged from the single path bitwise \
                 (max |diff| {})",
                max_diff(&refs[b], &out[b])
            );
        }
        // Determinism: an identical second run must be bitwise identical
        // (it rewrites the same kv slots with identical values).
        let a = engine.batch_decode_step(&reqs_at(0, &segs)).expect("run a");
        for i in 0..a.len() {
            assert!(
                bitwise(&a[i], &out[i]),
                "req {i}: B=20 double run diverged bitwise (max |diff| {})",
                max_diff(&a[i], &out[i])
            );
        }
        println!("b20: 20/20 bitwise == single references; double run bitwise identical");

        // Perf: per-request time at B=20 vs B=4 (acceptance: <= 2x).
        const N: usize = 50;
        let segs4 = &segs[..4];
        for s in 0..5 {
            engine.batch_decode_step(&reqs_at(s, &segs)).expect("b20 warmup");
        }
        let t0 = Instant::now();
        for s in 0..N {
            engine.batch_decode_step(&reqs_at(s, &segs)).expect("b20 step");
        }
        let dt20 = t0.elapsed();
        for s in 0..5 {
            engine.batch_decode_step(&reqs_at(s, segs4)).expect("b4 warmup");
        }
        let t1 = Instant::now();
        for s in 0..N {
            engine.batch_decode_step(&reqs_at(s, segs4)).expect("b4 step");
        }
        let dt4 = t1.elapsed();
        let per20 = dt20.as_secs_f64() * 1000.0 / N as f64 / toks.len() as f64;
        let per4 = dt4.as_secs_f64() * 1000.0 / N as f64 / segs4.len() as f64;
        println!(
            "b20 perf: {N} steps x B=20 in {dt20:?} ({:.2} ms/step, {per20:.3} ms/req), \
             {N} steps x B=4 in {dt4:?} ({:.2} ms/step, {per4:.3} ms/req) -> {:.2}x \
             per-request (acceptance <= 2.0x)",
            dt20.as_secs_f64() * 1000.0 / N as f64,
            dt4.as_secs_f64() * 1000.0 / N as f64,
            per20 / per4
        );
        assert!(
            per20 <= 2.0 * per4,
            "B=20 per-request {per20:.3}ms > 2x the B=4 per-request {per4:.3}ms"
        );
    }
}
