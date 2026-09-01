//! 006 T3E / S1-3 / Graph V2: decode-step graph integration at the engine
//! level (real machine).
//!
//! Gates:
//! 1. Wiring: with a real `GraphPool`, the first entry of each seq_len
//!    bucket attempts a capture of the decode launches with the full
//!    step declaration (Graph V2 `GraphStepDecl` — EVERY decode node is a
//!    `CustomKernel`: the small fused kernels, the two jgemm phase
//!    kernels per m=1 projection GEMM (`gemv_m1_f16f32` +
//!    `gemv_m1_f16f32_reduce` — the JitGemm path is loaded by default),
//!    and the flash decode kernel with its real 512-thread /
//!    `(d + max_kv) * 4`-smem geometry). No cublas node is left in the
//!    step, so the 13.x V2 read-back is never needed and every node is
//!    refresh-safe by construction. Captures succeed when the captured
//!    kernel-node count matches the declared specs (`graph_captures()`).
//!    Replay then runs bit-identical to eager on runtimes >= 13.x (the V2
//!    `cudaGraphNodeSetParams` refresh path — `graph_replays() > 0`); on
//!    12.x runtimes the CUkernel SetParams is rejected, replay fails
//!    closed and every step counts as an eager fallback (the documented
//!    Graph V2 boundary). The tests gate on `graph::runtime_version()`.
//! 2. Bitwise invariance: a graph-on engine and a graph-off engine produce
//!    bit-identical per-step logits and an identical greedy text (graph
//!    integration must never change single-step numerics — the replay
//!    path is only acceptable if it reproduces the eager kernels'
//!    deterministic fixed-order results exactly).
//! 3. Cost: the host-wall tpot of the graph-on engine must drop below the
//!    eager tpot, and the GPU-busy mean (cudaEvent pair on the engine's
//!    decode stream) is recorded against the wall mean — the replay path
//!    removes the per-kernel launch overhead, so wall approaches busy.
//!
//! Run (real machine; 13.2 nvcc mandatory — 12.6 cubins are all-zero):
//! ```text
//! REINFER_CUDA_NVCC=/usr/local/cuda-13.2/bin/nvcc \
//! REINFER_MODEL_DIR=/home/dora/.reinfer/models/Qwen/Qwen3-0.6B \
//! cargo test -p reinfer-cuda --features cuda --test graph_engine -- \
//!     --ignored --test-threads=1 --nocapture
//! ```
//!
//! Machine runtime note (driver 595.84 / CUDA 13.2): the V2 node
//! parameters APIs are only usable when libcudart.so.13 owns the
//! process's *global runtime scope*; this machine's binary links
//! libcudart.so.12, so the 13.x replay path requires
//! `LD_PRELOAD=/usr/local/cuda-13.2/lib64/libcudart.so.13.2.51` (plus
//! `LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64`). Without it the tests
//! exercise the 12.x fail-closed contract instead of the replay path.

#![cfg(feature = "cuda")]
#![allow(clippy::unwrap_used)] // test assertions panic on failure
#![allow(clippy::print_stdout)] // smoke output

mod gpu {
    use reinfer_core::DeviceId;
    use reinfer_cuda::engine::{Engine, argmax_first};
    use reinfer_cuda::graph::{GraphPool, v2_params_available};
    use reinfer_cuda::{CudaContext, CudaEvent, CudaStream};
    use std::path::PathBuf;
    use std::time::Instant;

    /// Context init; `None` when no usable GPU (skip on non-GPU machines).
    fn setup() -> Option<(CudaContext, DeviceId)> {
        let ctx = match CudaContext::init(DeviceId::new(0)) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("graph_engine: no GPU (skip): {e}");
                return None;
            }
        };
        let devid = ctx.device_id();
        let stream = CudaStream::new(devid).unwrap();
        let _ = stream.synchronize().unwrap();
        Some((ctx, devid))
    }

    fn cache_dir(tag: &str) -> Option<PathBuf> {
        Some(std::env::temp_dir().join(format!("reinfer-jit-{tag}")))
    }

    /// Deterministic LCG (mirror of the fmha_prefill harness).
    struct Lcg(u64);

    impl Lcg {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            self.0 >> 33
        }
    }

    /// Deterministic prompt ids (small alphabet, repeated tokens, valid ids).
    fn prompt_ids(seq: usize) -> Vec<u32> {
        let mut rng = Lcg(0x243f_6a88_85a3_08d3);
        (0..seq).map(|_| (rng.next_u64() as u32) % 4096 + 3).collect()
    }

    /// Dense per-token prefill (bypasses the FMHA selector — the test stays
    /// hermetic wrt the tune db) + `n` greedy (t=0) tokens. Returns the
    /// per-step logits (prefill steps first), the token sequence and the
    /// per-decode-step mean wall time (ms) over the `n` greedy steps.
    fn run(eng: &mut Engine, ids: &[u32], n: usize) -> (Vec<Vec<f32>>, Vec<u32>, f64) {
        let mut logits = Vec::new();
        for (i, &t) in ids.iter().enumerate() {
            logits.push(eng.step(t, i, i + 1).unwrap());
        }
        let mut toks = Vec::with_capacity(n);
        let mut cur = *ids.last().unwrap();
        let mut pos = ids.len();
        let t0 = Instant::now();
        while toks.len() < n {
            let lg = eng.step(cur, pos, pos + 1).unwrap();
            logits.push(lg.clone());
            let next = argmax_first(&lg);
            toks.push(next);
            cur = next;
            pos += 1;
        }
        let tpot = t0.elapsed().as_secs_f64() / n as f64 * 1000.0;
        (logits, toks, tpot)
    }

    /// GPU-busy mean (ms/step): a cudaEvent pair on the engine's decode
    /// stream around an `n`-token decode loop. The engine synchronizes at
    /// every step (the logits D2H readback), so the stream is idle at both
    /// record points and the elapsed time is the sum of the per-step GPU
    /// durations (plus tiny inter-step gaps) — the GPU-side cost of a
    /// step, measured against the host-wall `tpot` from `run`. Called
    /// after a warm-up run, so every seq_len bucket is already captured
    /// (the replay path is what is measured).
    fn gpu_busy_ms(eng: &mut Engine, ids: &[u32], n: usize) -> f64 {
        let stream = eng.decode_stream().clone();
        let t0 = CudaEvent::new(stream.device()).unwrap();
        let t1 = CudaEvent::new(stream.device()).unwrap();
        t0.record(&stream).unwrap();
        let mut cur = *ids.last().unwrap();
        let mut pos = ids.len();
        for _ in 0..n {
            let lg = eng.step(cur, pos, pos + 1).unwrap();
            cur = argmax_first(&lg);
            pos += 1;
        }
        t1.record(&stream).unwrap();
        t1.synchronize().unwrap();
        t0.elapsed_ms(&t1).unwrap() as f64 / n as f64
    }

    /// First divergent step index and the max logits delta there (both
    /// bit and magnitude) — diagnostic for replay-vs-eager divergence.
    fn first_divergence(logits_on: &[Vec<f32>], logits_off: &[Vec<f32>]) -> Option<(usize, f32, u64)> {
        for (i, (a, b)) in logits_on.iter().zip(logits_off.iter()).enumerate() {
            if a.len() != b.len() {
                return Some((i, f32::NAN, u64::MAX));
            }
            let mut maxd = 0.0f32;
            let mut bits: u64 = 0;
            for (x, y) in a.iter().zip(b.iter()) {
                if x.to_bits() != y.to_bits() {
                    bits += 1;
                    maxd = maxd.max((x - y).abs());
                }
            }
            if bits > 0 {
                return Some((i, maxd, bits));
            }
        }
        None
    }

    fn assert_bitwise(logits_on: &[Vec<f32>], logits_off: &[Vec<f32>], tag: &str) {
        assert_eq!(logits_on.len(), logits_off.len(), "{tag}: step count mismatch");
        let mut mismatches = 0usize;
        for (i, (a, b)) in logits_on.iter().zip(logits_off.iter()).enumerate() {
            assert_eq!(a.len(), b.len(), "{tag}: step {i}: logits len");
            for (x, y) in a.iter().zip(b.iter()) {
                if x.to_bits() != y.to_bits() {
                    mismatches += 1;
                }
            }
        }
        assert_eq!(mismatches, 0, "{tag}: logits must be bit-identical");
    }

    /// Graph on vs off: 128 decode tokens with bit-identical per-step logits
    /// and identical greedy text. The graph-on engine reports its graph
    /// counters: with the Graph V2 declaration the captures succeed per
    /// bucket; on 13.x the replays then serve every step bit-identically
    /// (`graph_replays() > 0`, eager fallbacks = capture attempts only) —
    /// on 12.x replay fails closed (CUkernel SetParams rejected) and every
    /// step counts as an eager fallback (the documented boundary).
    #[test]
    #[ignore = "gpu.yml: graph-smoke"]
    fn graph_on_vs_off_bitwise_and_text() {
        let Some((_ctx, devid)) = setup() else { return };
        let model_dir = match std::env::var_os("REINFER_MODEL_DIR").map(PathBuf::from) {
            Some(m) => m,
            None => {
                eprintln!("graph_engine: REINFER_MODEL_DIR unset (skip)");
                return;
            }
        };
        let arch = reinfer_cuda::arch::resolve_arch().unwrap();
        let cache = cache_dir("graph-smoke");
        let ids = prompt_ids(32);

        // Graph-off engine (REINFER_GRAPH=off equivalent: disabled pool).
        let mut off = Engine::load_with_graph(
            devid,
            &arch,
            cache.clone(),
            &model_dir,
            4096,
            GraphPool::disabled(),
        )
        .unwrap();
        // Graph-on engine (real pool: per-bucket capture attempts with the
        // full step declaration).
        let mut on = Engine::load_with_graph(
            devid,
            &arch,
            cache.clone(),
            &model_dir,
            4096,
            GraphPool::new(devid),
        )
        .unwrap();

        let (off_logits, off_toks, _off_tpot) = run(&mut off, &ids, 128);
        let (on_logits, on_toks, on_tpot) = run(&mut on, &ids, 128);

        assert_eq!(on_toks, off_toks, "graph on/off must produce identical text");
        assert_bitwise(&on_logits, &off_logits, "graph on/off");
        assert_eq!(off.graph_captures(), 0, "graph-off engine never captures");
        assert_eq!(off.graph_eager_fallbacks(), 0, "graph-off engine never falls back");
        // Runtime-adaptive (Graph V2): with the V2 node-params API in the
        // process's global scope (libcudart.so.13 linked or LD_PRELOADed)
        // every node is a custom kernel with full slot coverage, so
        // replays serve the steps (eager fallbacks = the one-time capture
        // attempt per new bucket); without it the linked 12.x setter
        // rejects the CUkernel — replay fails closed and every step
        // counts as an eager fallback. Gate on `v2_params_available`
        // (dlsym-based), NOT `cudaRuntimeGetVersion`: the linked symbol
        // is version-pinned to the build-time libcudart and reports 12.x
        // even when libcudart.so.13 is preloaded.
        if v2_params_available() {
            assert!(
                on.graph_replays() > 0,
                "graph-smoke: V2 must replay the all-custom graph"
            );
            assert!(
                on.graph_eager_fallbacks() <= 32,
                "graph-smoke: eager fallbacks must be capture attempts only (got {})",
                on.graph_eager_fallbacks()
            );
        } else {
            assert_eq!(on.graph_replays(), 0, "graph-smoke: 12.x must fail closed");
            assert!(on.graph_eager_fallbacks() > 0, "graph-smoke: 12.x must serve eager");
        }
        println!(
            "graph-smoke: {} steps bit-identical, {} tokens; graph-on: captures {}, \
             replays {}, eager fallbacks {}, tpot {on_tpot:.3} ms/step",
            on_logits.len(),
            on_toks.len(),
            on.graph_captures(),
            on.graph_replays(),
            on.graph_eager_fallbacks()
        );
    }

    /// Graph V2 anchor: pins the replay path at the engine level. On 13.x
    /// runtimes the all-custom declaration replays every captured step
    /// bit-identically to eager (`graph_replays() > 0`; eager fallbacks =
    /// one-time capture attempts per bucket), the wall tpot must drop
    /// below the eager tpot, and the GPU-busy mean (cudaEvent pair on the
    /// engine's stream, after warm-up so every bucket is captured) is
    /// recorded against the wall mean. On 12.x runtimes the CUkernel
    /// SetParams is rejected — replay fails closed and every step counts
    /// as an eager fallback (the documented Graph V2 boundary).
    #[test]
    #[ignore = "gpu.yml: graph-replay-anchor"]
    fn graph_capture_ok_replay_fail_closed_bitwise() {
        let Some((_ctx, devid)) = setup() else { return };
        let model_dir = match std::env::var_os("REINFER_MODEL_DIR").map(PathBuf::from) {
            Some(m) => m,
            None => {
                eprintln!("graph_engine: REINFER_MODEL_DIR unset (skip)");
                return;
            }
        };
        let arch = reinfer_cuda::arch::resolve_arch().unwrap();
        let cache = cache_dir("graph-anchor");
        let ids = prompt_ids(32);

        let mut off = Engine::load_with_graph(
            devid,
            &arch,
            cache.clone(),
            &model_dir,
            4096,
            GraphPool::disabled(),
        )
        .unwrap();
        let mut on = Engine::load_with_graph(
            devid,
            &arch,
            cache.clone(),
            &model_dir,
            4096,
            GraphPool::new(devid),
        )
        .unwrap();

        let (off_logits, off_toks, off_tpot) = run(&mut off, &ids, 128);
        let (on_logits, on_toks, on_tpot) = run(&mut on, &ids, 128);

        if let Some((i, maxd, bits)) = first_divergence(&on_logits, &off_logits) {
            eprintln!(
                "anchor: first divergence at step {i} (max logits delta {maxd:.6}, {bits} bit \
                 mismatches); on tok {:?} vs off tok {:?}",
                on_toks.get(i.wrapping_sub(ids.len())),
                off_toks.get(i.wrapping_sub(ids.len()))
            );
        }
        assert_eq!(on_toks, off_toks, "anchor: identical greedy text");
        assert_bitwise(&on_logits, &off_logits, "anchor on/off");
        assert_eq!(off.graph_captures(), 0, "anchor: graph-off never captures");
        assert_eq!(off.graph_replays(), 0, "anchor: graph-off never replays");
        assert_eq!(off.graph_eager_fallbacks(), 0, "anchor: graph-off never falls back");
        if v2_params_available() {
            // Graph V2 replay path (V2 node-params in the global scope —
            // libcudart.so.13 linked or LD_PRELOADed): replays serve the
            // steps bit-identically; eager fallbacks are the one-time
            // capture attempt per new bucket only (128 decode steps span
            // ~16 buckets). The wall mean must beat the eager mean — the
            // replay removes the per-kernel launch overhead — and the
            // GPU-busy mean (measured after warm-up, so every bucket is
            // already captured) is the floor the wall mean approaches.
                assert!(
                    on.graph_replays() > 0,
                    "anchor: 13.x must replay the all-custom graph"
                );
                assert!(
                    on.graph_eager_fallbacks() <= 32,
                    "anchor: eager fallbacks must be capture attempts only (got {})",
                    on.graph_eager_fallbacks()
                );
                assert!(
                    on_tpot < off_tpot,
                    "anchor: replay wall tpot {on_tpot:.3} must beat eager {off_tpot:.3} ms/step"
                );
                let on_busy = gpu_busy_ms(&mut on, &ids, 64);
                let off_busy = gpu_busy_ms(&mut off, &ids, 64);
                // Same kernels, same buffers — the GPU cost of a step must
                // be the same whether launched eagerly or replayed.
                assert!(
                    on_busy <= off_busy * 1.5 + 0.05,
                    "anchor: replay GPU busy {on_busy:.3} must match eager {off_busy:.3} ms/step"
                );
                println!(
                    "graph-anchor (Graph V2 replay): captures {}, replays {}, eager \
                     fallbacks {}; wall tpot on {on_tpot:.3} vs off {off_tpot:.3} ms/step \
                     (ratio {:.3}); GPU busy on {on_busy:.3} vs off {off_busy:.3} ms/step \
                     (wall/busy on: {:.3})",
                    on.graph_captures(),
                    on.graph_replays(),
                    on.graph_eager_fallbacks(),
                    on_tpot / off_tpot,
                    on_tpot / on_busy
                );
        } else {
            // 12.x fail-closed boundary: no replay may succeed; every step
            // serves eager (and counts as an eager fallback).
            assert_eq!(on.graph_replays(), 0, "anchor: 12.x must fail closed (replays 0)");
            assert!(on.graph_eager_fallbacks() > 0, "anchor: 12.x must serve eager");
            println!(
                "graph-anchor (12.x fail-closed): captures {}, replays {}, eager \
                 fallbacks {}; tpot on {on_tpot:.3} vs off {off_tpot:.3} ms/step \
                 (ratio {:.3})",
                on.graph_captures(),
                on.graph_replays(),
                on.graph_eager_fallbacks(),
                on_tpot / off_tpot
            );
        }
    }
}
