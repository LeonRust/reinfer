//! 014 L3 parity harness — Qwen3-0.6B F16 vs llama.cpp CPU referee (libllama).
//!
//! Four parity tiers (specs/014 plan.md D4/D6, r2): ① tokenizer 100% ·
//! ② F16 same-compute-type greedy tokens 100% (fallback record tier: sampled
//! per-step logits rel drift <= 1e-4) · ④ logits rel drift <= 1e-2 recorded.
//!
//! Protocol: plain-text completion on both sides (no chat template), the same
//! raw token sequence fed to both. The referee is the pinned llama.cpp CPU
//! build via bench/referee/llama_referee.cpp (greedy sampler, add_special=
//! true / parse_special=false — Qwen3 add_bos=false, so no BOS on either
//! side; first-max tie-break, same rule as the engine argmax_first). Both
//! sides generate exactly N_STEPS greedy tokens, no EOS stop.
//!
//! Engine driving (Backend level): row0 = `prefill_batch` returned logits
//! (position S-1, predicting the first generated token — same as the llama.cpp
//! prefill row), then decode steps at pos = S-1+i, kv_len = S+i (same slot
//! layout as llama.cpp). NOTE: `Engine::generate` / the run CLI re-feed the
//! last prompt token as the first decode input (slots S-1 and S both hold it
//! — an off-by-one vs llama.cpp; recorded in bench/notes.md 014 parity); the
//! harness therefore drives the engine with the aligned step loop instead.
//!
//! Drift metrics (relatively normalized to the row max — a plain relative
//! ratio blows up at near-zero logits): rel = |e-r| / max(|e|_max, |r|_max).
//! Two scopes: (a) trajectory same-prefix steps (identical contexts, gate
//! scope — meaningful while the two greedy streams agree), (b) conditional
//! full-64 steps (engine fed the referee tokens — pure-math closeness,
//! record scope).
//!
//! Env:
//!   REINFER_MODEL_DIR        HF safetensors model dir (existing convention)
//!   REINFER_REFEREE          path to the llama-referee binary
//!   REINFER_REFEREE_GGUF     Qwen3-0.6B GGUF for the referee
//!   REINFER_REFEREE_THREADS  referee CPU threads (default 8)
//!
//! Run (RTX 5090 laptop, sm_120a; nvcc 13.2 JIT rule):
//!   REINFER_CUDA_NVCC=/usr/local/cuda-13.2/bin/nvcc CUDA_VISIBLE_DEVICES=0 \
//!   REINFER_MODEL_DIR=$HOME/.reinfer/models/Qwen/Qwen3-0.6B \
//!   REINFER_REFEREE=/home/dora/Dev/ai-tokens/llama.cpp/build/bin/llama-referee \
//!   REINFER_REFEREE_GGUF=/home/dora/Dev/ai-tokens/bench-tmp-llamacpp/Qwen3-0.6B-f16.gguf \
//!   cargo test -p reinfer-cuda --features cuda --test parity -- --ignored --test-threads=1 --nocapture
//!
//! Hard asserts: logits fully finite (before any comparison — guards the
//! "both NaN → same argmax" false pass), vocab size equality, tier-①
//! tokenizer 100%. Tier ②/④ are reported record tiers (014 r2 allows the
//! record tier when 100% is not reached; the engine's f16-intermediate
//! attention is expected to keep logits within ~1e-2 rel of the llama.cpp
//! f32 path).

#![cfg(feature = "cuda")]
#![allow(clippy::unwrap_used)] // test assertions crash on failure
#![allow(clippy::print_stdout)] // parity report output

mod gpu {
    use reinfer_core::DeviceId;
    use reinfer_cuda::engine::{Engine, argmax_first};
    use reinfer_cuda::{CudaContext, CudaStream};
    use reinfer_tokenizer::Tokenizer;
    use std::io::Write;
    use std::path::{Path, PathBuf};
    use std::process::{Command, Stdio};

    /// Fixed 10-prompt set (same texts as bench-vs-vllm/prompts.py — raw
    /// completion input, no chat template). p8 contains U+00A0 (NBSP).
    const PROMPTS: &[(&str, &str)] = &[
        ("p0", "Hello"),
        ("p1", "The capital of France is"),
        ("p2", "写一首关于秋天的短诗"),
        ("p3", "Explain the difference between TCP and UDP in two sentences."),
        ("p4", "def fibonacci(n):\n    return"),
        ("p5", "9.11 和 9.9 哪个更大？请回答并解释。"),
        ("p6", "What is 17 * 23? Give the answer and a one-line explanation."),
        ("p7", "Summarize the concept of a database index in plain words."),
        ("p8", "    four-space and whitespace sensitive\u{a0}line"),
        ("p9", "列举三种排序算法并比较复杂度。"),
    ];

    /// Greedy tokens per prompt on each side.
    const N_STEPS: usize = 64;

    /// Pseudo-random positions per step for the sampled drift metric (fixed
    /// seed → reproducible; plus both argmax positions always included).
    const SAMPLE_COUNT: usize = 2048;
    const SAMPLE_SEED: u64 = 0x0140_1401_4014_0140;

    // ---------------- env plumbing ----------------

    fn env_path(name: &str) -> PathBuf {
        PathBuf::from(std::env::var(name).unwrap_or_else(|_| panic!("{name} must be set")))
    }

    fn model_dir() -> PathBuf {
        env_path("REINFER_MODEL_DIR")
    }

    fn referee_bin() -> PathBuf {
        env_path("REINFER_REFEREE")
    }

    fn referee_gguf() -> PathBuf {
        env_path("REINFER_REFEREE_GGUF")
    }

    fn referee_threads() -> usize {
        std::env::var("REINFER_REFEREE_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(8)
    }

    // ---------------- referee binary protocol ----------------

    struct RefStep {
        token: u32,
        logits: Vec<f32>,
    }

    struct RefRun {
        n_vocab: usize,
        prompt_ids: Vec<u32>,
        steps: Vec<RefStep>,
    }

    /// Spawn llama-referee with `prompt` on stdin; parse OUT.bin (layout in
    /// bench/referee/llama_referee.cpp).
    fn run_referee(prompt: &str, n_steps: usize) -> RefRun {
        let bin = referee_bin();
        let gguf = referee_gguf();
        let threads = referee_threads();
        let out = std::env::temp_dir().join(format!("reinfer-parity-{}.bin", std::process::id()));
        let out_s = out.to_string_lossy().into_owned();
        let mut child = Command::new(&bin)
            .args([
                "-m",
                gguf.to_str().unwrap(),
                "-n",
                &n_steps.to_string(),
                "-t",
                &threads.to_string(),
                "-o",
                &out_s,
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap_or_else(|e| panic!("spawn referee {bin:?}: {e}"));
        // Drop the pipe right after writing → referee sees EOF.
        child.stdin.take().unwrap().write_all(prompt.as_bytes()).unwrap();
        let cap = child.wait_with_output().expect("referee wait");
        let err = String::from_utf8_lossy(&cap.stderr);
        assert!(cap.status.success(), "referee failed: {err}");
        let data = std::fs::read(&out).unwrap_or_else(|e| panic!("read referee out {out_s}: {e}"));
        let _ = std::fs::remove_file(&out);
        parse_ref_out(&data, &err)
    }

    fn parse_ref_out(data: &[u8], err: &str) -> RefRun {
        let mut off = 0usize;
        let rd = |off: &mut usize| -> u32 {
            let v = u32::from_le_bytes(data[*off..*off + 4].try_into().unwrap());
            *off += 4;
            v
        };
        assert_eq!(rd(&mut off), 0x5041_5250, "referee magic (bad binary? {err})");
        assert_eq!(rd(&mut off), 1, "referee version");
        let n_vocab = rd(&mut off) as usize;
        let n_prompt = rd(&mut off) as usize;
        let prompt_ids = (0..n_prompt).map(|_| rd(&mut off)).collect::<Vec<_>>();
        let n_steps = rd(&mut off) as usize;
        let mut steps = Vec::with_capacity(n_steps);
        for _ in 0..n_steps {
            let token = rd(&mut off);
            let mut logits = Vec::with_capacity(n_vocab);
            for _ in 0..n_vocab {
                let b: [u8; 4] = data[off..off + 4].try_into().unwrap();
                off += 4;
                logits.push(f32::from_le_bytes(b));
            }
            steps.push(RefStep { token, logits });
        }
        assert_eq!(off, data.len(), "referee binary trailing bytes");
        RefRun { n_vocab, prompt_ids, steps }
    }

    // ---------------- engine plumbing ----------------

    fn load_engine_and_tok(model_dir: &Path) -> (Engine, Tokenizer) {
        let ctx = CudaContext::init(DeviceId::new(0)).expect("cuda ctx");
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
        (engine, tokenizer)
    }

    // ---------------- drift metrics ----------------

    /// SplitMix64 stream (no external deps; fixed seed → reproducible sample).
    fn splitmix64(x: u64) -> u64 {
        let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Fixed pseudo-random positions over the vocab plus extra positions
    /// (this step's argmax on both sides).
    fn sampled_positions(n_vocab: usize, extra: &[usize]) -> Vec<usize> {
        let mut pos: Vec<usize> = Vec::with_capacity(SAMPLE_COUNT + extra.len());
        let mut s = SAMPLE_SEED;
        for _ in 0..SAMPLE_COUNT {
            s = splitmix64(s);
            pos.push((s % n_vocab as u64) as usize);
        }
        pos.extend_from_slice(extra);
        pos.sort_unstable();
        pos.dedup();
        pos
    }

    /// Max relative logit drift over the given positions, normalized to the
    /// row max (|e-r| / max(|e|_max, |r|_max)) — well-conditioned at near-zero
    /// logits, unlike a plain relative ratio.
    fn rel_drift(e: &[f32], r: &[f32], positions: &[usize]) -> f32 {
        let rowmax = e.iter().chain(r.iter()).map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-12);
        positions.iter().map(|&i| (e[i] - r[i]).abs() / rowmax).fold(0.0f32, f32::max)
    }

    /// Full-vocab variant of `rel_drift` (strictly stronger than sampled).
    fn rel_drift_full(e: &[f32], r: &[f32]) -> f32 {
        let rowmax = e.iter().chain(r.iter()).map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-12);
        e.iter().zip(r.iter()).map(|(&a, &b)| (a - b).abs() / rowmax).fold(0.0f32, f32::max)
    }

    // ---------------- tier ①: tokenizer 100% ----------------

    #[test]
    #[ignore = "gpu.yml: l3-parity"]
    fn parity_tokenizer_tier1() {
        let dir = model_dir();
        let tok: serde_json::Value = serde_json::from_slice(
            &std::fs::read(dir.join("tokenizer.json")).expect("tokenizer.json"),
        )
        .expect("tokenizer json");
        let tcfg: serde_json::Value = serde_json::from_slice(
            &std::fs::read(dir.join("tokenizer_config.json")).expect("tokenizer_config.json"),
        )
        .expect("tokcfg json");
        let tokenizer = Tokenizer::from_hf_json(&tok, &tcfg).expect("hf tokenizer");

        let mut total = 0usize;
        let mut ok = 0usize;
        for (pid, text) in PROMPTS {
            let ids = tokenizer.encode(text, false).expect("encode");
            assert!(!ids.is_empty(), "{pid}: empty encode");
            let run = run_referee(text, 0);
            println!(
                "{pid}: engine encode({text:?}) -> {} tokens; referee -> {} tokens",
                ids.len(),
                run.prompt_ids.len()
            );
            assert_eq!(
                ids, run.prompt_ids,
                "tier ① tokenizer mismatch on {pid} (engine vs llama.cpp)"
            );
            ok += 1;
            total += ids.len();
        }
        println!("T1 tokenizer 100%: {ok}/{} prompts, {total} prompt tokens — PASS", PROMPTS.len());
    }

    // ---------------- tier ②: F16 greedy tokens + drift records ----------------

    struct PromptResult {
        id: &'static str,
        matched: usize,
        first_diff: Option<usize>,
        sampled_drift: f32,
        rel_drift_same_prefix: f32,
        rel_drift_cond_all: f32,
    }

    #[test]
    #[ignore = "gpu.yml: l3-parity"]
    fn parity_f16_tier2_generation() {
        let dir = model_dir();
        let (mut engine, tokenizer) = load_engine_and_tok(&dir);
        let vocab = engine.config().vocab_size;
        let mut results = Vec::new();
        let mut total_matched = 0usize;

        for (pid, text) in PROMPTS {
            // --- tier ① inside the generation flow (same referee spawn) ---
            let run = run_referee(text, N_STEPS);
            assert_eq!(run.n_vocab, vocab, "{pid}: referee n_vocab vs engine vocab_size");
            assert_eq!(run.steps.len(), N_STEPS, "{pid}: referee step count");

            let ids = tokenizer.encode(text, false).expect("encode");
            assert!(!ids.is_empty(), "{pid}: empty encode");
            assert_eq!(
                ids, run.prompt_ids,
                "tier ① tokenizer mismatch on {pid} (engine vs llama.cpp)"
            );
            let s = ids.len();

            // --- pass A: conditional drift (engine fed the referee tokens —
            // identical contexts for all 64 steps; pure-math closeness).
            // The second prefill_batch rewrites slots 0..S-1 idempotently and
            // stale slots beyond the attended range are masked by lens.
            let row0 = engine.prefill_batch(&ids).expect("prefill");
            assert!(
                row0.iter().all(|l| l.is_finite()),
                "{pid}: engine logits non-finite at prefill"
            );
            let mut cond_full_max = 0.0f32;
            for i in 0..N_STEPS {
                let (lg, ref_lg) = if i == 0 {
                    (row0.clone(), &run.steps[0].logits)
                } else {
                    let lg = engine
                        .step(run.steps[i - 1].token, s - 1 + i, s + i)
                        .expect("cond decode step");
                    (lg, &run.steps[i].logits)
                };
                assert!(
                    lg.iter().all(|l| l.is_finite()),
                    "{pid}: engine logits non-finite at step {i}"
                );
                assert_eq!(lg.len(), vocab, "{pid}: engine logits length");
                cond_full_max = cond_full_max.max(rel_drift_full(&lg, ref_lg));
            }

            // --- pass B: independent greedy trajectories (each side feeds its
            // own argmax — same-prefix drift is measured while they agree) ---
            let row0 = engine.prefill_batch(&ids).expect("prefill (pass B)");
            let mut matched = 0usize;
            let mut first_diff: Option<usize> = None;
            // (step, engine argmax, referee argmax, engine logits at step) —
            // kept for the near-tie report at the divergence point.
            let mut div: Option<(usize, u32, u32, Vec<f32>)> = None;
            let mut sampled_max = 0.0f32;
            let mut rel_same_prefix = 0.0f32;
            let mut eng_toks = Vec::with_capacity(N_STEPS);
            let mut ref_toks = Vec::with_capacity(N_STEPS);
            let mut cur = argmax_first(&row0);
            let r0_tok = run.steps[0].token;
            eng_toks.push(cur);
            ref_toks.push(r0_tok);
            if cur == r0_tok {
                matched += 1;
            } else {
                first_diff = Some(0);
            }
            if first_diff.is_none() {
                let s_pos = sampled_positions(vocab, &[cur as usize, r0_tok as usize]);
                sampled_max = sampled_max.max(rel_drift(&row0, &run.steps[0].logits, &s_pos));
                rel_same_prefix = rel_same_prefix.max(rel_drift_full(&row0, &run.steps[0].logits));
            }
            for i in 1..N_STEPS {
                let lg = engine.step(cur, s - 1 + i, s + i).expect("decode step");
                assert!(
                    lg.iter().all(|l| l.is_finite()),
                    "{pid}: engine logits non-finite at step {i}"
                );
                assert_eq!(lg.len(), vocab, "{pid}: engine logits length");
                let e_tok = argmax_first(&lg);
                let r_tok = run.steps[i].token;
                eng_toks.push(e_tok);
                ref_toks.push(r_tok);
                if e_tok == r_tok {
                    matched += 1;
                } else if first_diff.is_none() {
                    first_diff = Some(i);
                    div = Some((i, e_tok, r_tok, lg.clone()));
                }
                if first_diff.is_none() {
                    let ref_lg = &run.steps[i].logits;
                    let s_pos = sampled_positions(vocab, &[e_tok as usize, r_tok as usize]);
                    sampled_max = sampled_max.max(rel_drift(&lg, ref_lg, &s_pos));
                    rel_same_prefix = rel_same_prefix.max(rel_drift_full(&lg, ref_lg));
                }
                cur = e_tok;
            }

            if let Some((d, e_tok, r_tok, e_lg)) = div {
                // Near-tie report: window around the first divergence plus the
                // logit margins on both sides at the flipping position.
                let lo = d.saturating_sub(3);
                let hi = (d + 6).min(N_STEPS);
                let r_lg = &run.steps[d].logits;
                let e_delta = e_lg[e_tok as usize] - e_lg[r_tok as usize];
                let r_delta = r_lg[e_tok as usize] - r_lg[r_tok as usize];
                println!(
                    "{pid}: DIFF at step {d} (matched {matched}/{N_STEPS}) — record tier, \
                     sampled rel drift {sampled_max:.2e} / full {rel_same_prefix:.2e}",
                );
                println!(
                    "  flip: engine argmax {e_tok} vs referee argmax {r_tok} — \
                     engine margin {e_delta:+.3} / referee margin {r_delta:+.3} (logit units)"
                );
                println!("  window engine  [{lo}..{hi}): {:?}", &eng_toks[lo..hi]);
                println!("  window referee [{lo}..{hi}): {:?}", &ref_toks[lo..hi]);
            } else {
                println!("{pid}: {N_STEPS}/{N_STEPS} matched (tier ② 100%)");
            }
            total_matched += matched;
            results.push(PromptResult {
                id: pid,
                matched,
                first_diff,
                sampled_drift: sampled_max,
                rel_drift_same_prefix: rel_same_prefix,
                rel_drift_cond_all: cond_full_max,
            });
        }

        println!();
        println!(
            "{:<8} {:>6} {:>6} {:>10} {:>13} {:>13} {:>13}",
            "prompt",
            "tokens",
            "match",
            "first_diff",
            "sampled_drift",
            "rel_same_pfx",
            "rel_cond_64"
        );
        for r in &results {
            println!(
                "{:<8} {N_STEPS:>6} {:>6} {:>10?} {:>13.2e} {:>13.2e} {:>13.2e}",
                r.id,
                r.matched,
                r.first_diff,
                r.sampled_drift,
                r.rel_drift_same_prefix,
                r.rel_drift_cond_all
            );
        }
        let total = PROMPTS.len() * N_STEPS;
        let pct = total_matched as f64 / total as f64 * 100.0;
        let cond_max = results.iter().map(|r| r.rel_drift_cond_all).fold(0.0f32, f32::max);
        println!();
        if total_matched == total {
            println!("T2 F16 greedy: {total_matched}/{total} (100%) — GATE MET");
        } else {
            println!(
                "T2 F16 greedy: {total_matched}/{total} ({pct:.1}%) — RECORD TIER \
                 (fallback sampled drift <= 1e-4: see per-prompt sampled_drift)",
            );
        }
        println!(
            "T4 logits rel drift (full vocab, conditional 64 steps): max {cond_max:.2e} \
             (record, <= 1e-2)",
        );
    }
}
