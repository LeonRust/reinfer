//! S1-7 FMHA heuristics microbenchmark (real machine).
//!
//! Kernel-latency table for every variant set (v0..v3, see fmha.rs
//! `FmhaVariant` / kernels/fmha_kernels.cu) across prefill seqlen shapes,
//! timed with cudaEvent on the launch stream (per-launch GPU time, no host
//! synchronize in the timed section).
//!
//! Run (real machine; 13.2 nvcc mandatory):
//! ```text
//! REINFER_CUDA_NVCC=/usr/local/cuda-13.2/bin/nvcc \
//! cargo test -p reinfer-cuda --features cuda --test fmha_heuristics_bench -- \
//!     --ignored --test-threads=1 --nocapture
//! ```

#![cfg(feature = "cuda")]
#![allow(clippy::unwrap_used)] // test assertions panic on failure
#![allow(clippy::print_stdout)] // benchmark table output

mod gpu {
    use reinfer_core::DeviceId;
    use reinfer_cuda::buffer::{DeviceBuffer, HostBuffer, MemRef, copy};
    use reinfer_cuda::engine::Engine;
    use reinfer_cuda::event::CudaEvent;
    use reinfer_cuda::fmha::FmhaKernels;
    use reinfer_cuda::{CudaContext, CudaStream};
    use reinfer_tokenizer::Tokenizer;
    use std::io::Write;
    use std::path::PathBuf;
    use std::process::{Command, Stdio};

    /// Context init; `None` when no usable GPU (skip on non-GPU machines).
    fn setup() -> Option<(CudaContext, u32, DeviceId)> {
        let ctx = match CudaContext::init(DeviceId::new(0)) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("fmha_heuristics_bench: no GPU (skip): {e}");
                return None;
            }
        };
        let devid = ctx.device_id();
        let dev = devid.index();
        let stream = CudaStream::new(devid).unwrap();
        let _ = stream.synchronize().unwrap();
        Some((ctx, dev, devid))
    }

    /// Dense-test Lcg replica (see fmha_prefill.rs for the reference impl).
    struct Lcg(u64);

    impl Lcg {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            self.0 >> 33
        }
    }

    /// Deterministic pseudo-random f16 values (LCG bits — no NaN/Inf/denorm).
    fn fill_f16(buf: &mut [u16]) {
        let mut s: u64 = 0x9e3779b97f4a7c15;
        for b in buf.iter_mut() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            // f16 in [0.5, 2.0): exponent 0x3800 | mantissa bits.
            *b = 0x3800 | ((s >> 42) as u16 & 0x3ff);
        }
    }

    /// Probe the device's dynamic-smem opt-in ceiling (diagnostic; the v3
    /// variant's 128 KiB request was rejected with INVALID_VALUE).
    #[test]
    #[ignore]
    fn smem_optin_ceiling() {
        let Some((_ctx, dev, devid)) = setup() else { return };
        let stream = CudaStream::new(devid).unwrap();
        let arch = reinfer_cuda::arch::resolve_arch().unwrap();
        let fmha = FmhaKernels::new(&arch, None, stream.clone()).unwrap();
        let mut lo = 98304u32;
        let mut hi = 262144u32;
        // Binary search the max accepted bytes on the v3 (BM=256) kernel.
        while hi - lo > 1024 {
            let mid = (lo + hi) / 2;
            match fmha.probe_set_max_smem(3, mid) {
                Ok(()) => lo = mid,
                Err(_) => hi = mid,
            }
        }
        println!("device max dynamic smem opt-in ≈ {lo} B on {arch}");
        let _ = dev;
    }

    #[test]
    #[ignore]
    fn fmha_variant_latency_table() {
        let Some((_ctx, dev, devid)) = setup() else { return };
        let stream = CudaStream::new(devid).unwrap();
        let arch = reinfer_cuda::arch::resolve_arch().unwrap();
        // Same cache as the engine (default dir) — the 16-symbol cubin is
        // compiled once and shared with the CLI runs.
        let fmha = FmhaKernels::new(&arch, None, stream.clone()).unwrap();
        let heads = 16u32;
        let kv_heads = 8u32;
        let d = 128u32;
        let nqk = heads * d; // 2048
        let kvk = kv_heads * d; // 1024
        // Even and odd shapes: the engine's real prompts are rarely a
        // multiple of 128, and odd shapes take the boundary-checked kernel
        // symbols (a different, slower code path).
        let seqs = [256u32, 512, 1024, 2048, 4096, 255, 1023, 2047, 4095];
        let iters = 24u32;

        println!("FMHA variant latency (us, median of {iters} launches, causal, GQA {heads}/{kv_heads}, d={d}):");
        println!("  seq    |  v0 128x128w4 |  v1 128x128w8 |  v2 128x64w4 |  v3 256x128w8");
        for &seq in &seqs {
            let sq_r = seq.div_ceil(256) * 256; // max block_m 256 covers all variants
            let n = (seq * nqk) as usize;
            let nk = (seq * kvk) as usize;
            let q = DeviceBuffer::alloc(devid, n * 2).unwrap();
            let k = DeviceBuffer::alloc(devid, nk * 2).unwrap();
            let v = DeviceBuffer::alloc(devid, nk * 2).unwrap();
            let o = DeviceBuffer::alloc(devid, n * 2).unwrap();
            let lse = DeviceBuffer::alloc(devid, (heads as usize) * (sq_r as usize) * 4).unwrap();
            // Deterministic contents (values irrelevant — latency only).
            let hq = HostBuffer::alloc(n * 2).unwrap();
            let hk = HostBuffer::alloc(nk * 2).unwrap();
            fill_f16(unsafe { std::slice::from_raw_parts_mut(hq.as_ptr() as *mut u8 as *mut u16, n) });
            fill_f16(unsafe { std::slice::from_raw_parts_mut(hk.as_ptr() as *mut u8 as *mut u16, nk) });
            copy(&mut MemRef::Device(&q), &MemRef::Host(&hq), n * 2, None).unwrap();
            copy(&mut MemRef::Device(&k), &MemRef::Host(&hk), nk * 2, None).unwrap();
            copy(&mut MemRef::Device(&v), &MemRef::Host(&hk), nk * 2, None).unwrap();

            let mut row = String::from("  ");
            row.push_str(&format!("{:>4}  |", seq));
            for vi in 0..fmha.variants().len() {
                let ev0 = CudaEvent::new(devid).unwrap();
                let ev1 = CudaEvent::new(devid).unwrap();
                let launch = |fmha: &FmhaKernels| {
                    fmha.launch_batched_prefill_variant(
                        dev,
                        vi,
                        q.as_ptr() as *const u16,
                        k.as_ptr() as *const u16,
                        v.as_ptr() as *const u16,
                        o.as_ptr() as *mut u16,
                        lse.as_ptr() as *mut f32,
                        seq,
                        1,
                        heads,
                        kv_heads,
                        d,
                    )
                    .unwrap()
                };
                println!("  -- launching seq={seq} variant v{vi} (warmup)");
                for _ in 0..2 {
                    launch(&fmha); // warmup
                }
                stream.synchronize().unwrap();
                let mut samples = Vec::with_capacity(iters as usize);
                for _ in 0..iters {
                    ev0.record(&stream).unwrap();
                    launch(&fmha);
                    ev1.record(&stream).unwrap();
                    ev1.synchronize().unwrap();
                    samples.push(ev0.elapsed_ms(&ev1).unwrap() * 1000.0);
                }
                samples.sort_by(|a, b| a.total_cmp(b));
                let med = samples[samples.len() / 2];
                row.push_str(&format!(" {:>11.1} |", med));
            }
            println!("{row}");
        }
    }

    /// Variant numeric identity (S1-7): all variants must agree with the
    /// baseline v0 within D7 fp16 (1 ulp). Same input, causal, odd seqlen
    /// (exercises the partial-block path in every variant).
    #[test]
    #[ignore]
    fn fmha_variant_numeric_identity() {
        let Some((_ctx, dev, devid)) = setup() else { return };
        let stream = CudaStream::new(devid).unwrap();
        let arch = reinfer_cuda::arch::resolve_arch().unwrap();
        let fmha = FmhaKernels::new(&arch, None, stream.clone()).unwrap();
        // Two input regimes: all-positive f16 in [0.5, 2.0) (engine-like
        // magnitudes) and signed gaussian (the fmha_prefill dense-reference
        // input class — QK^T scores then straddle zero, so many P values
        // quantize to f16 0).
        for (heads, kv_heads, signed) in
            [(16u32, 8u32, false), (8u32, 4u32, true), (16u32, 8u32, true), (8u32, 4u32, false)]
        {
            let d = 128u32;
            let nqk = (heads * d) as usize;
            let kvk = (kv_heads * d) as usize;
            for &seq in &[256u32, 1024, 2047] {
                let sq_r = (seq as usize).div_ceil(256) * 256;
                let n = seq as usize * nqk;
                let nk = seq as usize * kvk;
                let q = DeviceBuffer::alloc(devid, n * 2).unwrap();
                let k = DeviceBuffer::alloc(devid, nk * 2).unwrap();
                let v = DeviceBuffer::alloc(devid, nk * 2).unwrap();
                let o = DeviceBuffer::alloc(devid, n * 2).unwrap();
                let lse = DeviceBuffer::alloc(devid, heads as usize * sq_r * 4).unwrap();
                let hq = HostBuffer::alloc(n * 2).unwrap();
                let hk = HostBuffer::alloc(nk * 2).unwrap();
                if signed {
                    // EXACT replica of the fmha_prefill dense-reference
                    // generator (same Lcg seed formula, Box-Muller, f16
                    // truncation; Q scaled by 1/sqrt(d) as the engine
                    // pre-scales). The dense test's seq=256 draw makes v2
                    // output 0.0 at row 128 while v0 stays correct — this
                    // probe must reproduce that draw bit-for-bit.
                    let mut rng = Lcg(0x9e37_79b9_7f4a_7c15 ^ ((seq as u64) << 32) ^ 1);
                    let mut gauss = || -> f32 {
                        let u1 = (rng.next_u64() as f64 / (1u64 << 31) as f64).max(1e-12);
                        let u2 = rng.next_u64() as f64 / (1u64 << 31) as f64;
                        let r = (-2.0 * u1.ln()).sqrt();
                        (r * (2.0 * std::f64::consts::PI * u2).cos()) as f32
                    };
                    let f16t = |f: f32| -> u16 {
                        let b = f.to_bits();
                        let sgn = ((b >> 16) & 0x8000) as u16;
                        let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
                        if exp >= 31 {
                            return sgn | 0x7c00;
                        }
                        if exp <= 0 {
                            return sgn;
                        }
                        sgn | ((exp as u16) << 10) | ((b >> 13) as u16 & 0x3ff)
                    };
                    let scale = 1.0f32 / 128.0f32.sqrt();
                    let mut fill = |buf: &mut [u16], is_q: bool| {
                        for b in buf.iter_mut() {
                            let f = gauss();
                            *b = f16t(if is_q { f * scale } else { f });
                        }
                    };
                    fill(unsafe { std::slice::from_raw_parts_mut(hq.as_ptr() as *mut u8 as *mut u16, n) }, true);
                    fill(unsafe { std::slice::from_raw_parts_mut(hk.as_ptr() as *mut u8 as *mut u16, nk) }, false);
                } else {
                    fill_f16(unsafe { std::slice::from_raw_parts_mut(hq.as_ptr() as *mut u8 as *mut u16, n) });
                    fill_f16(unsafe { std::slice::from_raw_parts_mut(hk.as_ptr() as *mut u8 as *mut u16, nk) });
                }
                copy(&mut MemRef::Device(&q), &MemRef::Host(&hq), n * 2, None).unwrap();
                copy(&mut MemRef::Device(&k), &MemRef::Host(&hk), nk * 2, None).unwrap();
                copy(&mut MemRef::Device(&v), &MemRef::Host(&hk), nk * 2, None).unwrap();

                // Baseline v0 output.
                let run = |vi: usize| -> Vec<u16> {
                    fmha.launch_batched_prefill_variant(
                        dev, vi,
                        q.as_ptr() as *const u16,
                        k.as_ptr() as *const u16,
                        v.as_ptr() as *const u16,
                        o.as_ptr() as *mut u16,
                        lse.as_ptr() as *mut f32,
                        seq, 1, heads, kv_heads, d,
                    )
                    .unwrap();
                    stream.synchronize().unwrap();
                    let mut h = HostBuffer::alloc(n * 2).unwrap();
                    copy(&mut MemRef::Host(&mut h), &MemRef::Device(&o), n * 2, None).unwrap();
                    unsafe {
                        std::slice::from_raw_parts(h.as_ptr() as *const u16, n).to_vec()
                    }
                };
                let refo = run(0);
                for vi in 1..fmha.variants().len() {
                    let out = run(vi);
                    // fp16 D7 tier: |diff| <= 1 ulp (bit distance <= 1).
                    let mut worst = 0i32;
                    let mut nbad = 0usize;
                    for (a, b) in refo.iter().zip(out.iter()) {
                        let d = a.wrapping_sub(*b) as i16 as i32;
                        if d > worst {
                            worst = d;
                        }
                        if d.abs() > 1 {
                            nbad += 1;
                        }
                    }
                    println!(
                        "  seq={seq} h={heads} signed={signed} v{vi} vs v0: \
                         max_bit_diff={worst} elems_over_1ulp={nbad}/{}",
                        n
                    );
                    assert_eq!(
                        nbad,
                        0,
                        "variant {vi} diverges from v0 beyond 1 ulp at seq {seq} h={heads} signed={signed}"
                    );
                }
            }
        }
    }

    /// Context bisect: the dense-reference test (fmha_prefill.rs) fails with
    /// the v2 pick at seq=256 h=8 (FMHA O == 0.0 at row 128) while the
    /// identity test above passes with the identical launch — so the trigger
    /// is in the launch context, not the kernel/variant. The dense test
    /// differs by: (a) `DecodeKernels::new` runs first, (b) a per-test JIT
    /// cache dir, (c) a fresh stream. Probe all four combinations.
    #[test]
    #[ignore]
    fn fmha_variant_context_bisect() {
        let Some((_ctx, dev, devid)) = setup() else { return };
        let stream0 = CudaStream::new(devid).unwrap();
        let arch = reinfer_cuda::arch::resolve_arch().unwrap();
        let heads = 8u32;
        let kv_heads = 4u32;
        let d = 128u32;
        let seq = 256u32;
        let nqk = (heads * d) as usize;
        let kvk = (kv_heads * d) as usize;
        let n = seq as usize * nqk;
        let nk = seq as usize * kvk;

        // Dense-test draw (see the identity test above for the generator).
        let mut rng = Lcg(0x9e37_79b9_7f4a_7c15 ^ ((seq as u64) << 32) ^ 1);
        let mut gauss = || -> f32 {
            let u1 = (rng.next_u64() as f64 / (1u64 << 31) as f64).max(1e-12);
            let u2 = rng.next_u64() as f64 / (1u64 << 31) as f64;
            let r = (-2.0 * u1.ln()).sqrt();
            (r * (2.0 * std::f64::consts::PI * u2).cos()) as f32
        };
        let f16t = |f: f32| -> u16 {
            let b = f.to_bits();
            let sgn = ((b >> 16) & 0x8000) as u16;
            let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
            if exp >= 31 {
                return sgn | 0x7c00;
            }
            if exp <= 0 {
                return sgn;
            }
            sgn | ((exp as u16) << 10) | ((b >> 13) as u16 & 0x3ff)
        };
        let scale = 1.0f32 / 128.0f32.sqrt();
        let hq = HostBuffer::alloc(n * 2).unwrap();
        let hk = HostBuffer::alloc(nk * 2).unwrap();
        let hv = HostBuffer::alloc(nk * 2).unwrap();
        {
            let qb = unsafe { std::slice::from_raw_parts_mut(hq.as_ptr() as *mut u8 as *mut u16, n) };
            let kb = unsafe { std::slice::from_raw_parts_mut(hk.as_ptr() as *mut u8 as *mut u16, nk) };
            let vb = unsafe { std::slice::from_raw_parts_mut(hv.as_ptr() as *mut u8 as *mut u16, nk) };
            for b in qb.iter_mut() {
                *b = f16t(gauss() * scale);
            }
            for b in kb.iter_mut() {
                *b = f16t(gauss());
            }
            for b in vb.iter_mut() {
                *b = f16t(gauss()); // separate draw, exactly like the dense test
            }
        }

        // Device buffers (one set, reused across contexts; content is
        // identical in every leg).
        let q = DeviceBuffer::alloc(devid, n * 2).unwrap();
        let k = DeviceBuffer::alloc(devid, nk * 2).unwrap();
        let v = DeviceBuffer::alloc(devid, nk * 2).unwrap();
        let o = DeviceBuffer::alloc(devid, n * 2).unwrap();
        let lse = DeviceBuffer::alloc(devid, heads as usize * (seq as usize) * 4).unwrap();
        copy(&mut MemRef::Device(&q), &mut MemRef::Host(&hq), n * 2, None).unwrap();
        copy(&mut MemRef::Device(&k), &mut MemRef::Host(&hk), nk * 2, None).unwrap();
        copy(&mut MemRef::Device(&v), &mut MemRef::Host(&hv), nk * 2, None).unwrap();
        let f16_to_f32 = |bits: u16| {
            let sign = ((bits & 0x8000) as u32) << 16;
            let exp = (bits >> 10) & 0x1f;
            let man = (bits & 0x3ff) as u32;
            let bits32 = match exp {
                0 if man == 0 => sign,
                0 => (sign | (man << 13)) as u32 | (127 - 15) << 23,
                0x1f if man == 0 => sign | 0x7f80_0000,
                0x1f => sign | 0x7f80_0000 | (man << 13),
                e => (sign | ((e as u32 + (127 - 15)) << 23) | (man << 13)),
            };
            f32::from_bits(bits32)
        };
        // Context order matters: contexts run sequentially in ONE process and
        // only the first one sees a COLD CUDA context (first kernel launch
        // ever). Put (true,false) first — the dense test's exact cache dir
        // without decode — so a cold v2 first-launch must reproduce there if
        // the cubin is innocent; then the warm-context legs act as controls.
        for (with_cache, with_decode) in [(true, false), (false, false), (true, true), (false, true)] {
            let stream = if with_decode {
                CudaStream::new(devid).unwrap()
            } else {
                stream0.clone()
            };
            let cache: Option<PathBuf> = if with_cache {
                Some(std::env::temp_dir().join("reinfer-jit-fmha-prefill"))
            } else {
                None
            };
            if with_decode {
                // Same load as the dense test (decode cubin first).
                reinfer_cuda::decode::DecodeKernels::new(&arch, cache.clone(), stream.clone())
                    .unwrap();
            }
            let fmha = FmhaKernels::new(&arch, cache, stream.clone()).unwrap();
            let run = |fmha: &FmhaKernels, vi: usize| -> Vec<u16> {
                fmha.launch_batched_prefill_variant(
                    dev, vi,
                    q.as_ptr() as *const u16,
                    k.as_ptr() as *const u16,
                    v.as_ptr() as *const u16,
                    o.as_ptr() as *mut u16,
                    lse.as_ptr() as *mut f32,
                    seq, 1, heads, kv_heads, d,
                )
                .unwrap();
                stream.synchronize().unwrap();
                let mut h = HostBuffer::alloc(n * 2).unwrap();
                copy(&mut MemRef::Host(&mut h), &mut MemRef::Device(&o), n * 2, None).unwrap();
                unsafe { std::slice::from_raw_parts(h.as_ptr() as *const u16, n).to_vec() }
            };
            // v2 FIRST in each context: the dense test launches v2 as the very
            // first kernel on its context (decode kernels are loaded, never
            // launched) — every earlier probe launched v0 before v2.
            let o2 = run(&fmha, 2);
            let o0 = run(&fmha, 0);
            let r0 = f16_to_f32(o0[128 * nqk]);
            let r2 = f16_to_f32(o2[128 * nqk]);
            let mut nbad = 0usize;
            for (a, b) in o0.iter().zip(o2.iter()) {
                if (a.wrapping_sub(*b) as i16 as i32).abs() > 1 {
                    nbad += 1;
                }
            }
            println!(
                "  ctx(cache={with_cache} decode={with_decode}): v0 O[128,0,0]={r0:.4e} \
                 v2 O[128,0,0]={r2:.4e} nbad={nbad}"
            );
        }
    }

    /// f16 bits -> f32 (exact).
    fn f16_to_f32(bits: u16) -> f32 {
        let sign = ((bits & 0x8000) as u32) << 16;
        let exp = (bits >> 10) & 0x1f;
        let man = (bits & 0x3ff) as u32;
        let bits32 = match exp {
            0 if man == 0 => sign,
            0 => (sign | (man << 13)) as u32 | (127 - 15) << 23,
            0x1f if man == 0 => sign | 0x7f80_0000,
            0x1f => sign | 0x7f80_0000 | (man << 13),
            e => (sign | ((e as u32 + (127 - 15)) << 23) | (man << 13)),
        };
        f32::from_bits(bits32)
    }

    /// Dense-test input replica (fmha_prefill.rs): gaussian f16 q/k/v drawn
    /// from the dense Lcg with the (seq, batch=1) seed, q pre-scaled by
    /// 1/sqrt(d). Returns (q, k, v) HostBuffers.
    fn dense_draw(seq: u32, heads: u32, kv_heads: u32) -> (HostBuffer, HostBuffer, HostBuffer) {
        let d = 128u32;
        let n = (seq * heads * d) as usize;
        let nk = (seq * kv_heads * d) as usize;
        let mut rng = Lcg(0x9e37_79b9_7f4a_7c15 ^ ((seq as u64) << 32) ^ 1);
        let mut gauss = || -> f32 {
            let u1 = (rng.next_u64() as f64 / (1u64 << 31) as f64).max(1e-12);
            let u2 = rng.next_u64() as f64 / (1u64 << 31) as f64;
            let r = (-2.0 * u1.ln()).sqrt();
            (r * (2.0 * std::f64::consts::PI * u2).cos()) as f32
        };
        let f16t = |f: f32| -> u16 {
            let b = f.to_bits();
            let sgn = ((b >> 16) & 0x8000) as u16;
            let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
            if exp >= 31 {
                return sgn | 0x7c00;
            }
            if exp <= 0 {
                return sgn;
            }
            sgn | ((exp as u16) << 10) | ((b >> 13) as u16 & 0x3ff)
        };
        let scale = 1.0f32 / 128.0f32.sqrt();
        let hq = HostBuffer::alloc(n * 2).unwrap();
        let hk = HostBuffer::alloc(nk * 2).unwrap();
        let hv = HostBuffer::alloc(nk * 2).unwrap();
        {
            let qb = unsafe { std::slice::from_raw_parts_mut(hq.as_ptr() as *mut u8 as *mut u16, n) };
            let kb = unsafe { std::slice::from_raw_parts_mut(hk.as_ptr() as *mut u8 as *mut u16, nk) };
            let vb = unsafe { std::slice::from_raw_parts_mut(hv.as_ptr() as *mut u8 as *mut u16, nk) };
            for b in qb.iter_mut() {
                *b = f16t(gauss() * scale);
            }
            for b in kb.iter_mut() {
                *b = f16t(gauss());
            }
            for b in vb.iter_mut() {
                *b = f16t(gauss()); // separate draw, exactly like the dense test
            }
        }
        (hq, hk, hv)
    }

    /// The dense gate's draw (fmha_prefill.rs `fmha_vs_dense_case`): Lcg
    /// seed `0x9e37_79b9_7f4a_7c15 ^ ((seq as u64) << 32) ^ (batch as u64)`;
    /// q is drawn unscaled, truncated to f16, widened to f32, THEN scaled
    /// — the gate scales after the f16 conversion. Same draw for every
    /// (batch, head) slot: the row is picked per (b,h) on the host.
    fn gate_draw(seq: u32, batch: u32) -> (HostBuffer, HostBuffer, HostBuffer) {
        let d = 128u32;
        let n = (seq * 8 * d) as usize;
        let nk = (seq * 4 * d) as usize;
        let b = batch as usize;
        let mut rng = Lcg(0x9e37_79b9_7f4a_7c15 ^ ((seq as u64) << 32) ^ (batch as u64));
        let mut gauss = || -> f32 {
            let u1 = (rng.next_u64() as f64 / (1u64 << 31) as f64).max(1e-12);
            let u2 = rng.next_u64() as f64 / (1u64 << 31) as f64;
            let r = (-2.0 * u1.ln()).sqrt();
            (r * (2.0 * std::f64::consts::PI * u2).cos()) as f32
        };
        // The gate's truncating f32->f16 (keeps subnormals, unlike dense_draw).
        let gate_f16 = |f: f32| -> u16 {
            let bits = f.to_bits();
            let sign = ((bits >> 16) & 0x8000) as u16;
            let exp = ((bits >> 23) & 0xff) as i32;
            let man = bits & 0x7f_ffff;
            if exp == 0xff {
                return sign | 0x7c00 | ((man >> 13) & 0x3ff) as u16;
            }
            let half_exp = exp - 127 + 15;
            if half_exp <= 0 {
                if half_exp < -10 {
                    return sign;
                }
                let subm = (man | 0x800_000) >> (1 - half_exp + 13);
                return sign | subm as u16;
            }
            if half_exp >= 31 {
                return sign | 0x7c00;
            }
            sign | ((half_exp as u16) << 10) | ((man >> 13) as u16)
        };
        let scale = 1.0f32 / 128.0f32.sqrt();
        let hq = HostBuffer::alloc(n * b * 2).unwrap();
        let hk = HostBuffer::alloc(nk * b * 2).unwrap();
        let hv = HostBuffer::alloc(nk * b * 2).unwrap();
        {
            let qb = unsafe {
                std::slice::from_raw_parts_mut(hq.as_ptr() as *mut u8 as *mut u16, n * b)
            };
            let kb = unsafe {
                std::slice::from_raw_parts_mut(hk.as_ptr() as *mut u8 as *mut u16, nk * b)
            };
            let vb = unsafe {
                std::slice::from_raw_parts_mut(hv.as_ptr() as *mut u8 as *mut u16, nk * b)
            };
            // q: draw unscaled -> f16 -> f32 -> scale -> f16 (gate order).
            for x in qb.iter_mut() {
                *x = gate_f16(f16_to_f32(gate_f16(gauss())) * scale);
            }
            for x in kb.iter_mut() {
                *x = gate_f16(gauss());
            }
            for x in vb.iter_mut() {
                *x = gate_f16(gauss());
            }
        }
        (hq, hk, hv)
    }

    /// v2 cold-context race (S1-7 bisect conclusion): v2 (128x64x4w) outputs
    /// 0.0 for Q block 1 (row 128) when it is the FIRST kernel launch on a
    /// fresh CUDA context; on a warm context it is bit-identical to v0. This
    /// test is one process = one cold context, replicating the dense test's
    /// exact launch (decode loaded-but-never-launched, fmha from the
    /// fmha-prefill cache, v2 via the pick path) — must show the 0.0.
    #[test]
    #[ignore]
    fn fmha_v2_cold_first_launch() {
        let Some((_ctx, dev, devid)) = setup() else { return };
        let stream = CudaStream::new(devid).unwrap();
        let arch = reinfer_cuda::arch::resolve_arch().unwrap();
        let cache = std::env::temp_dir().join("reinfer-jit-fmha-prefill");
        reinfer_cuda::decode::DecodeKernels::new(&arch, Some(cache.clone()), stream.clone())
            .unwrap();
        let fmha = FmhaKernels::new(&arch, Some(cache), stream.clone()).unwrap();
        let (heads, kv_heads, seq) = (8u32, 4u32, 256u32);
        let d = 128u32;
        let nqk = (heads * d) as usize;
        let n = (seq * heads * d) as usize;
        let nk = (seq * kv_heads * d) as usize;
        let (hq, hk, hv) = dense_draw(seq, heads, kv_heads);
        let q = DeviceBuffer::alloc(devid, n * 2).unwrap();
        let k = DeviceBuffer::alloc(devid, nk * 2).unwrap();
        let v = DeviceBuffer::alloc(devid, nk * 2).unwrap();
        let o = DeviceBuffer::alloc(devid, n * 2).unwrap();
        let lse = DeviceBuffer::alloc(devid, heads as usize * (seq as usize) * 4).unwrap();
        copy(&mut MemRef::Device(&q), &mut MemRef::Host(&hq), n * 2, None).unwrap();
        copy(&mut MemRef::Device(&k), &mut MemRef::Host(&hk), nk * 2, None).unwrap();
        copy(&mut MemRef::Device(&v), &mut MemRef::Host(&hv), nk * 2, None).unwrap();
        // v2 via the pick path — the FIRST launch on this context.
        fmha.launch_batched_prefill(
            dev,
            q.as_ptr() as *const u16,
            k.as_ptr() as *const u16,
            v.as_ptr() as *const u16,
            o.as_ptr() as *mut u16,
            lse.as_ptr() as *mut f32,
            seq,
            1,
            heads,
            kv_heads,
            d,
        )
        .unwrap();
        stream.synchronize().unwrap();
        let mut h = HostBuffer::alloc(n * 2).unwrap();
        copy(&mut MemRef::Host(&mut h), &mut MemRef::Device(&o), n * 2, None).unwrap();
        let bits = unsafe { std::slice::from_raw_parts(h.as_ptr() as *const u16, n) };
        println!(
            "cold-first v2 (pick): O[128,0,0]={:.4e} (0.0e0 = race reproduced)",
            f16_to_f32(bits[128 * nqk])
        );
        // LSE diagnostic: the kernel's early-exit path (n_block_max <=
        // n_block_min) writes O=0 AND LSE=+inf for the block's rows — if
        // LSE[128] = +inf, the bidm=1 CTA took the early-exit branch.
        let mut hlse = HostBuffer::alloc((heads as usize) * (seq as usize) * 4).unwrap();
        copy(
            &mut MemRef::Host(&mut hlse),
            &mut MemRef::Device(&lse),
            (heads as usize) * (seq as usize) * 4,
            None,
        )
        .unwrap();
        let lsef: &[f32] =
            unsafe { std::slice::from_raw_parts(hlse.as_ptr() as *const f32, heads as usize * seq as usize) };
        println!(
            "cold-first v2 LSE: [0..4]={:?} [128..132]={:?} (inf = early-exit branch)",
            &lsef[..4],
            &lsef[128..132]
        );
        // Second v2 launch, same handle/stream: in the bisect this still
        // failed — a failed v2 does not heal the context.
        fmha.launch_batched_prefill(
            dev,
            q.as_ptr() as *const u16,
            k.as_ptr() as *const u16,
            v.as_ptr() as *const u16,
            o.as_ptr() as *mut u16,
            lse.as_ptr() as *mut f32,
            seq,
            1,
            heads,
            kv_heads,
            d,
        )
        .unwrap();
        stream.synchronize().unwrap();
        copy(&mut MemRef::Host(&mut h), &mut MemRef::Device(&o), n * 2, None).unwrap();
        let bits2 = unsafe { std::slice::from_raw_parts(h.as_ptr() as *const u16, n) };
        println!(
            "v2 #2 (same process): O[128,0,0]={:.4e}",
            f16_to_f32(bits2[128 * nqk])
        );
        // Heal check: a v0 launch (which always works), then v2 again.
        let launch_v2 = |fmha: &FmhaKernels| {
            fmha.launch_batched_prefill(
                dev,
                q.as_ptr() as *const u16,
                k.as_ptr() as *const u16,
                v.as_ptr() as *const u16,
                o.as_ptr() as *mut u16,
                lse.as_ptr() as *mut f32,
                seq,
                1,
                heads,
                kv_heads,
                d,
            )
            .unwrap();
        };
        let launch_v0 = |fmha: &FmhaKernels| {
            fmha.launch_batched_prefill_variant(
                dev,
                0,
                q.as_ptr() as *const u16,
                k.as_ptr() as *const u16,
                v.as_ptr() as *const u16,
                o.as_ptr() as *mut u16,
                lse.as_ptr() as *mut f32,
                seq,
                1,
                heads,
                kv_heads,
                d,
            )
            .unwrap();
        };
        launch_v0(&fmha);
        stream.synchronize().unwrap();
        launch_v2(&fmha);
        stream.synchronize().unwrap();
        copy(&mut MemRef::Host(&mut h), &mut MemRef::Device(&o), n * 2, None).unwrap();
        let bits3 = unsafe { std::slice::from_raw_parts(h.as_ptr() as *const u16, n) };
        println!(
            "v2 after v0 heal: O[128,0,0]={:.4e}",
            f16_to_f32(bits3[128 * nqk])
        );
    }

    /// Last-CTA anatomy (S1-7): at seq=512 the v2 kernel runs 4 m-block
    /// CTAs (blockIdx.x=0..3). Fill o+lse with 0xAAAA/0x41414141 via plain
    /// memcpy (no FMHA launch can fire a preflight), then ONE v2 launch
    /// via the pick path on a PRIMED key (preflight disarmed). Blocks that
    /// keep the pattern were never written by their CTA. Discriminates
    /// "only the last CTA (x=gM-1)" from "every x>=1 CTA".
    #[test]
    #[ignore]
    fn fmha_last_cta_anatomy() {
        let Some((_ctx, dev, devid)) = setup() else { return };
        let stream = CudaStream::new(devid).unwrap();
        let arch = reinfer_cuda::arch::resolve_arch().unwrap();
        let cache = std::env::temp_dir().join("reinfer-jit-fmha-prefill");
        let fmha = FmhaKernels::new(&arch, Some(cache), stream.clone()).unwrap();
        let (heads, kv_heads, seq) = (8u32, 4u32, 512u32);
        let d = 128u32;
        let nqk = (heads * d) as usize;
        let n = (seq * heads * d) as usize;
        let nk = (seq * kv_heads * d) as usize;
        let (hq, hk, hv) = dense_draw(seq, heads, kv_heads);
        let q = DeviceBuffer::alloc(devid, n * 2).unwrap();
        let k = DeviceBuffer::alloc(devid, nk * 2).unwrap();
        let v = DeviceBuffer::alloc(devid, nk * 2).unwrap();
        let o = DeviceBuffer::alloc(devid, n * 2).unwrap();
        let lse = DeviceBuffer::alloc(devid, heads as usize * (seq as usize) * 4).unwrap();
        copy(&mut MemRef::Device(&q), &mut MemRef::Host(&hq), n * 2, None).unwrap();
        copy(&mut MemRef::Device(&k), &mut MemRef::Host(&hk), nk * 2, None).unwrap();
        copy(&mut MemRef::Device(&v), &mut MemRef::Host(&hv), nk * 2, None).unwrap();
        // Prime: the first pick-path launch fires the v0 preflight on this
        // key; the key then stays primed for the pattern launch below.
        fmha.launch_batched_prefill(
            dev,
            q.as_ptr() as *const u16,
            k.as_ptr() as *const u16,
            v.as_ptr() as *const u16,
            o.as_ptr() as *mut u16,
            lse.as_ptr() as *mut f32,
            seq,
            1,
            heads,
            kv_heads,
            d,
        )
        .unwrap();
        stream.synchronize().unwrap();
        // Pattern-fill O and LSE via plain copies.
        let pat = HostBuffer::alloc(n * 2).unwrap();
        {
            let s: &mut [u16] = unsafe {
                std::slice::from_raw_parts_mut(pat.as_ptr() as *mut u16, n)
            };
            s.fill(0xAAAA);
        }
        let plse = HostBuffer::alloc(heads as usize * (seq as usize) * 4).unwrap();
        {
            let s: &mut [u32] = unsafe {
                std::slice::from_raw_parts_mut(plse.as_ptr() as *mut u32, (heads as usize) * (seq as usize))
            };
            s.fill(0x41414141);
        }
        copy(&mut MemRef::Device(&o), &mut MemRef::Host(&pat), n * 2, None).unwrap();
        copy(
            &mut MemRef::Device(&lse),
            &mut MemRef::Host(&plse),
            heads as usize * (seq as usize) * 4,
            None,
        )
        .unwrap();
        // One v2 launch (key primed — no preflight).
        fmha.launch_batched_prefill(
            dev,
            q.as_ptr() as *const u16,
            k.as_ptr() as *const u16,
            v.as_ptr() as *const u16,
            o.as_ptr() as *mut u16,
            lse.as_ptr() as *mut f32,
            seq,
            1,
            heads,
            kv_heads,
            d,
        )
        .unwrap();
        stream.synchronize().unwrap();
        let mut ho = HostBuffer::alloc(n * 2).unwrap();
        copy(&mut MemRef::Host(&mut ho), &mut MemRef::Device(&o), n * 2, None).unwrap();
        let ob: &[u16] = unsafe { std::slice::from_raw_parts(ho.as_ptr() as *const u16, n) };
        let mut hlse = HostBuffer::alloc(heads as usize * (seq as usize) * 4).unwrap();
        copy(
            &mut MemRef::Host(&mut hlse),
            &mut MemRef::Device(&lse),
            heads as usize * (seq as usize) * 4,
            None,
        )
        .unwrap();
        let lsef: &[u32] = unsafe {
            std::slice::from_raw_parts(hlse.as_ptr() as *const u32, (heads as usize) * (seq as usize))
        };
        let nblocks = (seq as usize) / 128;
        let mut o_stale = Vec::new();
        let mut lse_stale = Vec::new();
        for b in 0..nblocks {
            let mut oc = 0usize;
            for s in b * 128..(b + 1) * 128 {
                for h in 0..heads as usize {
                    let base = s * nqk + h * d as usize;
                    for i in 0..d as usize {
                        if ob[base + i] == 0xAAAA {
                            oc += 1;
                        }
                    }
                }
            }
            o_stale.push(oc);
            let mut lc = 0usize;
            for s in b * 128..(b + 1) * 128 {
                for h in 0..heads as usize {
                    if lsef[h * (seq as usize) + s] == 0x41414141 {
                        lc += 1;
                    }
                }
            }
            lse_stale.push(lc);
        }
        println!(
            "seq=512 pattern-fill: o-stale per 128-row block (rows b*128..): {:?} (full={}, target {}) | lse-stale per block: {:?} (full={})",
            o_stale,
            128 * nqk,
            128 * nqk,
            lse_stale,
            8 * 128
        );
        // Occupancy test: relaunch the v2 with its TRUE smem (65536 B)
        // instead of the over-declared 98304 B. More resident CTAs per SM
        // => the write-capable first wave covers more of the grid. If the
        // stale set shrinks (e.g. to block 3 only, or empty), the skip is
        // occupancy/wave-correlated; if it stays {2,3}, it is not.
        let mut test_smem = |variant: usize, smem: u32, label: &str| {
            copy(&mut MemRef::Device(&o), &mut MemRef::Host(&pat), n * 2, None).unwrap();
            copy(
                &mut MemRef::Device(&lse),
                &mut MemRef::Host(&plse),
                heads as usize * (seq as usize) * 4,
                None,
            )
            .unwrap();
            fmha.launch_batched_prefill_variant_smem(
                dev,
                variant,
                q.as_ptr() as *const u16,
                k.as_ptr() as *const u16,
                v.as_ptr() as *const u16,
                o.as_ptr() as *mut u16,
                lse.as_ptr() as *mut f32,
                seq,
                1,
                heads,
                kv_heads,
                d,
                Some(smem),
            )
            .unwrap();
            stream.synchronize().unwrap();
            copy(&mut MemRef::Host(&mut ho), &mut MemRef::Device(&o), n * 2, None).unwrap();
            let ob: &[u16] = unsafe { std::slice::from_raw_parts(ho.as_ptr() as *const u16, n) };
            let mut blocks = Vec::new();
            for b in 0..nblocks {
                let mut oc = 0usize;
                for s in b * 128..(b + 1) * 128 {
                    for h in 0..heads as usize {
                        let base = s * nqk + h * d as usize;
                        for i in 0..d as usize {
                            if ob[base + i] == 0xAAAA {
                                oc += 1;
                            }
                        }
                    }
                }
                blocks.push(oc);
            }
            println!("seq=512 {label} (smem={smem}): o-stale per block: {blocks:?}");
        };
        // v0-control FIRST: the v0 (128x128) writes all blocks; its RDEBUG
        // enter/exit prints show every x reaches the epilogue.
        test_smem(0, 98304, "v0-control");
        // v1-control: 8 warps, 128x128, declared smem == launch smem (98304,
        // no over-declaration). If the second-half drop is tied to the
        // declared-vs-launched smem mismatch (v2 declares 65536, launched at
        // 98304), the v1 must write ALL blocks — the engine pick could then
        // move to v1 (perf: v1 ~1.3x v2 per the heuristics table).
        test_smem(1, 98304, "v1-control");
        // v2 with the over-declared 98304 B (the engine's launch config):
        // if the second-half CTAs' stores are skipped, blocks {2,3} stay
        // stale; RDEBUG exit-prints show whether x=2,3 reach the epilogue.
        test_smem(2, 98304, "v2-98304-rerun");
        // LAST (faults with Err: Driver): v2 with its true 64 KiB smem.
        test_smem(2, 65536, "v2-true-smem");
    }

    /// Warmup cure probe (S1-7): a trivial kernel launch (v0 at seq=128 —
    /// grid (1,1,8)) before v2 on the cold context must make v2 correct.
    /// If this passes, the fix is a warmup launch at FmhaKernels::new.
    #[test]
    #[ignore]
    fn fmha_v2_cold_after_warmup() {
        let Some((_ctx, dev, devid)) = setup() else { return };
        let stream = CudaStream::new(devid).unwrap();
        let arch = reinfer_cuda::arch::resolve_arch().unwrap();
        let cache = std::env::temp_dir().join("reinfer-jit-fmha-prefill");
        let fmha = FmhaKernels::new(&arch, Some(cache), stream.clone()).unwrap();
        let (heads, kv_heads, seq) = (8u32, 4u32, 256u32);
        let d = 128u32;
        let nqk = (heads * d) as usize;
        let n = (seq * heads * d) as usize;
        let nk = (seq * kv_heads * d) as usize;
        let (hq, hk, hv) = dense_draw(seq, heads, kv_heads);
        let q = DeviceBuffer::alloc(devid, n * 2).unwrap();
        let k = DeviceBuffer::alloc(devid, nk * 2).unwrap();
        let v = DeviceBuffer::alloc(devid, nk * 2).unwrap();
        let o = DeviceBuffer::alloc(devid, n * 2).unwrap();
        let lse = DeviceBuffer::alloc(devid, heads as usize * (seq as usize) * 4).unwrap();
        copy(&mut MemRef::Device(&q), &mut MemRef::Host(&hq), n * 2, None).unwrap();
        copy(&mut MemRef::Device(&k), &mut MemRef::Host(&hk), nk * 2, None).unwrap();
        copy(&mut MemRef::Device(&v), &mut MemRef::Host(&hv), nk * 2, None).unwrap();
        // Warmup: v0 at seq=128 (grid (1,1,8)) — the first launch on this
        // context. Reads the first halves of q/k/v; output goes to o.
        fmha.launch_batched_prefill_variant(
            dev,
            0,
            q.as_ptr() as *const u16,
            k.as_ptr() as *const u16,
            v.as_ptr() as *const u16,
            o.as_ptr() as *mut u16,
            lse.as_ptr() as *mut f32,
            128,
            1,
            heads,
            kv_heads,
            d,
        )
        .unwrap();
        stream.synchronize().unwrap();
        // Now v2 via the pick path.
        fmha.launch_batched_prefill(
            dev,
            q.as_ptr() as *const u16,
            k.as_ptr() as *const u16,
            v.as_ptr() as *const u16,
            o.as_ptr() as *mut u16,
            lse.as_ptr() as *mut f32,
            seq,
            1,
            heads,
            kv_heads,
            d,
        )
        .unwrap();
        stream.synchronize().unwrap();
        let mut h = HostBuffer::alloc(n * 2).unwrap();
        copy(&mut MemRef::Host(&mut h), &mut MemRef::Device(&o), n * 2, None).unwrap();
        let bits = unsafe { std::slice::from_raw_parts(h.as_ptr() as *const u16, n) };
        let got = f16_to_f32(bits[128 * nqk]);
        println!("warmup-then-v2 cold context: O[128,0,0]={got:.4e} (-1.2091e-1 = cured)");
        assert!(
            (got + 1.2091e-1).abs() < 1e-3,
            "warmup did not cure the cold-context race: got {got}"
        );
    }

    /// Smem-declaration probe: v2's true smem need is 64 KiB (kSmemQ 32 KiB +
    /// kSmemKV 32 KiB); the launch over-declares 98304 B (same class as v0,
    /// one CTA/SM). Test whether the cold-first failure persists with the
    /// exact 65536 B declaration.
    #[test]
    #[ignore]
    fn fmha_v2_cold_smem_65536() {
        let Some((_ctx, dev, devid)) = setup() else { return };
        let stream = CudaStream::new(devid).unwrap();
        let arch = reinfer_cuda::arch::resolve_arch().unwrap();
        let cache = std::env::temp_dir().join("reinfer-jit-fmha-prefill");
        let fmha = FmhaKernels::new(&arch, Some(cache), stream.clone()).unwrap();
        let (heads, kv_heads, seq) = (8u32, 4u32, 256u32);
        let d = 128u32;
        let nqk = (heads * d) as usize;
        let n = (seq * heads * d) as usize;
        let nk = (seq * kv_heads * d) as usize;
        let (hq, hk, hv) = dense_draw(seq, heads, kv_heads);
        let q = DeviceBuffer::alloc(devid, n * 2).unwrap();
        let k = DeviceBuffer::alloc(devid, nk * 2).unwrap();
        let v = DeviceBuffer::alloc(devid, nk * 2).unwrap();
        let o = DeviceBuffer::alloc(devid, n * 2).unwrap();
        let lse = DeviceBuffer::alloc(devid, heads as usize * (seq as usize) * 4).unwrap();
        copy(&mut MemRef::Device(&q), &mut MemRef::Host(&hq), n * 2, None).unwrap();
        copy(&mut MemRef::Device(&k), &mut MemRef::Host(&hk), nk * 2, None).unwrap();
        copy(&mut MemRef::Device(&v), &mut MemRef::Host(&hv), nk * 2, None).unwrap();
        for (label, smem) in [("true-64KiB", Some(65536u32)), ("declared-96KiB", None)] {
            fmha.launch_batched_prefill_variant_smem(
                dev,
                2,
                q.as_ptr() as *const u16,
                k.as_ptr() as *const u16,
                v.as_ptr() as *const u16,
                o.as_ptr() as *mut u16,
                lse.as_ptr() as *mut f32,
                seq,
                1,
                heads,
                kv_heads,
                d,
                smem,
            )
            .unwrap();
            stream.synchronize().unwrap();
            let mut h = HostBuffer::alloc(n * 2).unwrap();
            copy(&mut MemRef::Host(&mut h), &mut MemRef::Device(&o), n * 2, None).unwrap();
            let bits = unsafe { std::slice::from_raw_parts(h.as_ptr() as *const u16, n) };
            println!(
                "cold-first v2 smem={label}: O[128,0,0]={:.4e}",
                f16_to_f32(bits[128 * nqk])
            );
        }
    }

    /// Cure-mechanism probe (S1-7): the constructor warmup (v0@256 on
    /// scratch buffers) does NOT cure the dense test's v2 — so the cure
    /// observed in fmha_v2_cold_first_launch (v0@256 with the REAL q/k/v/o/
    /// lse buffers, then v2 on the same buffers) must depend on the v0
    /// launch using the SAME buffers as the v2 launch. If this passes, the
    /// fix is a preflight v0 launch with the caller's own buffers right
    /// before the first picked launch.
    #[test]
    #[ignore]
    fn fmha_cure_probe_same_buffers_v0_256() {
        let Some((_ctx, dev, devid)) = setup() else { return };
        let stream = CudaStream::new(devid).unwrap();
        let arch = reinfer_cuda::arch::resolve_arch().unwrap();
        let cache = std::env::temp_dir().join("reinfer-jit-fmha-prefill");
        reinfer_cuda::decode::DecodeKernels::new(&arch, Some(cache.clone()), stream.clone())
            .unwrap();
        // Constructor warmup fires here (v0@256 on scratch buffers) — the
        // dense-test configuration that does NOT cure.
        let fmha = FmhaKernels::new(&arch, Some(cache), stream.clone()).unwrap();
        let (heads, kv_heads, seq) = (8u32, 4u32, 256u32);
        let d = 128u32;
        let nqk = (heads * d) as usize;
        let n = (seq * heads * d) as usize;
        let nk = (seq * kv_heads * d) as usize;
        let (hq, hk, hv) = dense_draw(seq, heads, kv_heads);
        let q = DeviceBuffer::alloc(devid, n * 2).unwrap();
        let k = DeviceBuffer::alloc(devid, nk * 2).unwrap();
        let v = DeviceBuffer::alloc(devid, nk * 2).unwrap();
        let o = DeviceBuffer::alloc(devid, n * 2).unwrap();
        let lse = DeviceBuffer::alloc(devid, heads as usize * (seq as usize) * 4).unwrap();
        copy(&mut MemRef::Device(&q), &mut MemRef::Host(&hq), n * 2, None).unwrap();
        copy(&mut MemRef::Device(&k), &mut MemRef::Host(&hk), nk * 2, None).unwrap();
        copy(&mut MemRef::Device(&v), &mut MemRef::Host(&hv), nk * 2, None).unwrap();
        // Interleave: v0@256 with the REAL buffers (as in the heal test),
        // then v2 with the same buffers.
        fmha.launch_batched_prefill_variant(
            dev,
            0,
            q.as_ptr() as *const u16,
            k.as_ptr() as *const u16,
            v.as_ptr() as *const u16,
            o.as_ptr() as *mut u16,
            lse.as_ptr() as *mut f32,
            seq,
            1,
            heads,
            kv_heads,
            d,
        )
        .unwrap();
        stream.synchronize().unwrap();
        fmha.launch_batched_prefill(
            dev,
            q.as_ptr() as *const u16,
            k.as_ptr() as *const u16,
            v.as_ptr() as *const u16,
            o.as_ptr() as *mut u16,
            lse.as_ptr() as *mut f32,
            seq,
            1,
            heads,
            kv_heads,
            d,
        )
        .unwrap();
        stream.synchronize().unwrap();
        let mut h = HostBuffer::alloc(n * 2).unwrap();
        copy(&mut MemRef::Host(&mut h), &mut MemRef::Device(&o), n * 2, None).unwrap();
        let bits = unsafe { std::slice::from_raw_parts(h.as_ptr() as *const u16, n) };
        let got = f16_to_f32(bits[128 * nqk]);
        println!("warmup+v0(same buffers)+v2: O[128,0,0]={got:.4e}");
        assert!(
            (got + 1.2091e-1).abs() < 1e-3,
            "v0 with the real buffers did not cure: got {got}"
        );
    }

    /// Cure-mechanism probe (S1-7), grid variant: the interleaved v0 uses
    /// seq=128 (grid (1,1,8) — NO bidm=1 CTA) with the REAL buffers. If v2
    /// is cured, the mechanism is buffer-related, not grid-related; if not,
    /// the cure requires a grid.x=2 launch.
    #[test]
    #[ignore]
    fn fmha_cure_probe_same_buffers_v0_128() {
        let Some((_ctx, dev, devid)) = setup() else { return };
        let stream = CudaStream::new(devid).unwrap();
        let arch = reinfer_cuda::arch::resolve_arch().unwrap();
        let cache = std::env::temp_dir().join("reinfer-jit-fmha-prefill");
        reinfer_cuda::decode::DecodeKernels::new(&arch, Some(cache.clone()), stream.clone())
            .unwrap();
        let fmha = FmhaKernels::new(&arch, Some(cache), stream.clone()).unwrap();
        let (heads, kv_heads, seq) = (8u32, 4u32, 256u32);
        let d = 128u32;
        let nqk = (heads * d) as usize;
        let n = (seq * heads * d) as usize;
        let nk = (seq * kv_heads * d) as usize;
        let (hq, hk, hv) = dense_draw(seq, heads, kv_heads);
        let q = DeviceBuffer::alloc(devid, n * 2).unwrap();
        let k = DeviceBuffer::alloc(devid, nk * 2).unwrap();
        let v = DeviceBuffer::alloc(devid, nk * 2).unwrap();
        let o = DeviceBuffer::alloc(devid, n * 2).unwrap();
        let lse = DeviceBuffer::alloc(devid, heads as usize * (seq as usize) * 4).unwrap();
        copy(&mut MemRef::Device(&q), &mut MemRef::Host(&hq), n * 2, None).unwrap();
        copy(&mut MemRef::Device(&k), &mut MemRef::Host(&hk), nk * 2, None).unwrap();
        copy(&mut MemRef::Device(&v), &mut MemRef::Host(&hv), nk * 2, None).unwrap();
        fmha.launch_batched_prefill_variant(
            dev,
            0,
            q.as_ptr() as *const u16,
            k.as_ptr() as *const u16,
            v.as_ptr() as *const u16,
            o.as_ptr() as *mut u16,
            lse.as_ptr() as *mut f32,
            128,
            1,
            heads,
            kv_heads,
            d,
        )
        .unwrap();
        stream.synchronize().unwrap();
        fmha.launch_batched_prefill(
            dev,
            q.as_ptr() as *const u16,
            k.as_ptr() as *const u16,
            v.as_ptr() as *const u16,
            o.as_ptr() as *mut u16,
            lse.as_ptr() as *mut f32,
            seq,
            1,
            heads,
            kv_heads,
            d,
        )
        .unwrap();
        stream.synchronize().unwrap();
        let mut h = HostBuffer::alloc(n * 2).unwrap();
        copy(&mut MemRef::Host(&mut h), &mut MemRef::Device(&o), n * 2, None).unwrap();
        let bits = unsafe { std::slice::from_raw_parts(h.as_ptr() as *const u16, n) };
        let got = f16_to_f32(bits[128 * nqk]);
        println!("warmup+v0@128(same buffers)+v2: O[128,0,0]={got:.4e}");
        println!("  (pass = buffer-related mechanism, no grid.x=2 needed)");
        assert!(
            (got + 1.2091e-1).abs() < 1e-3,
            "v0@128 with the real buffers did not cure: got {got}"
        );
    }

    /// Odd-MN probe (S1-7): the dense gate's (100,2) shape fails at row 64
    /// (the odd-MN second sub-block, grid.x=1 — no bidm=1 CTA) even on a
    /// warm context. Map the failure: batch=1 vs batch=2, v0 vs v2, and
    /// whether a real-buffer v0 immediately before cures it (the
    /// same-buffer rule from the even-MN race). NOTE: O row 64 of batch 0
    /// is element 64*(b*nqk) — reading 64*nqk is row 32 at batch=2 (an
    /// earlier probe bug that made (100,2) look unaffected).
    #[test]
    #[ignore]
    fn fmha_v2_odd_mn_probe() {
        let Some((_ctx, dev, devid)) = setup() else { return };
        let stream = CudaStream::new(devid).unwrap();
        let arch = reinfer_cuda::arch::resolve_arch().unwrap();
        let cache = std::env::temp_dir().join("reinfer-jit-fmha-prefill");
        let (heads, kv_heads) = (8u32, 4u32);
        let d = 128u32;
        let seq = 100u32;
        // Replicate the dense gate's warm-context situation: the gate runs
        // six shapes before (100,2), each with its own FmhaKernels (fresh
        // library/function handles on the same primary context). First
        // instance warms the context with a full (256,1) sequence.
        let fmha0 = FmhaKernels::new(&arch, Some(cache.clone()), stream.clone()).unwrap();
        {
            let (heads0, kv_heads0) = (8u32, 4u32);
            let n0 = (256 * heads0 * d) as usize;
            let nk0 = (256 * kv_heads0 * d) as usize;
            let (hq0, hk0, hv0) = dense_draw(256, heads0, kv_heads0);
            let q0 = DeviceBuffer::alloc(devid, n0 * 2).unwrap();
            let k0 = DeviceBuffer::alloc(devid, nk0 * 2).unwrap();
            let v0 = DeviceBuffer::alloc(devid, nk0 * 2).unwrap();
            let o0 = DeviceBuffer::alloc(devid, n0 * 2).unwrap();
            let lse0 =
                DeviceBuffer::alloc(devid, heads0 as usize * 256 * 4).unwrap();
            copy(&mut MemRef::Device(&q0), &mut MemRef::Host(&hq0), n0 * 2, None).unwrap();
            copy(&mut MemRef::Device(&k0), &mut MemRef::Host(&hk0), nk0 * 2, None).unwrap();
            copy(&mut MemRef::Device(&v0), &mut MemRef::Host(&hv0), nk0 * 2, None).unwrap();
            fmha0
                .launch_batched_prefill(
                    dev,
                    q0.as_ptr() as *const u16,
                    k0.as_ptr() as *const u16,
                    v0.as_ptr() as *const u16,
                    o0.as_ptr() as *mut u16,
                    lse0.as_ptr() as *mut f32,
                    256,
                    1,
                    heads0,
                    kv_heads0,
                    d,
                )
                .unwrap();
            stream.synchronize().unwrap();
        }
        // Second instance — fresh handles, warm context, FRESH stream (the
        // gate's case: each fmha_vs_dense_case creates a new stream).
        let stream1 = CudaStream::new(devid).unwrap();
        let fmha = FmhaKernels::new(&arch, Some(cache), stream1.clone()).unwrap();
        for batch in [2u32, 1] {
            let nqk = (heads * d) as usize;
            let n = (seq * heads * d) as usize;
            let nk = (seq * kv_heads * d) as usize;
            let b = batch as usize;
            // Dense-test replica draw (seed ^ batch).
            let mut rng = Lcg(0x9e37_79b9_7f4a_7c15 ^ ((seq as u64) << 32) ^ (batch as u64));
            let mut gauss = || -> f32 {
                let u1 = (rng.next_u64() as f64 / (1u64 << 31) as f64).max(1e-12);
                let u2 = rng.next_u64() as f64 / (1u64 << 31) as f64;
                let r = (-2.0 * u1.ln()).sqrt();
                (r * (2.0 * std::f64::consts::PI * u2).cos()) as f32
            };
            let f16t = |f: f32| -> u16 {
                let fb = f.to_bits();
                let sgn = ((fb >> 16) & 0x8000) as u16;
                let exp = ((fb >> 23) & 0xff) as i32 - 127 + 15;
                if exp >= 31 {
                    return sgn | 0x7c00;
                }
                if exp <= 0 {
                    return sgn;
                }
                sgn | ((exp as u16) << 10) | ((fb >> 13) as u16 & 0x3ff)
            };
            let scale = 1.0f32 / 128.0f32.sqrt();
            let hq = HostBuffer::alloc(n * b * 2).unwrap();
            let hk = HostBuffer::alloc(nk * b * 2).unwrap();
            let hv = HostBuffer::alloc(nk * b * 2).unwrap();
            {
                let qb = unsafe {
                    std::slice::from_raw_parts_mut(hq.as_ptr() as *mut u8 as *mut u16, n * b)
                };
                let kb = unsafe {
                    std::slice::from_raw_parts_mut(hk.as_ptr() as *mut u8 as *mut u16, nk * b)
                };
                let vb = unsafe {
                    std::slice::from_raw_parts_mut(hv.as_ptr() as *mut u8 as *mut u16, nk * b)
                };
                // Dense-test EXACT gen: draw unscaled -> f16 -> f32 -> scale
                // -> f16 (the gate's (100,2) failed; my scale-before-f16
                // probe data passed — test whether the rounding matters).
                for x in qb.iter_mut() {
                    *x = f16t(f16_to_f32(f16t(gauss())) * scale);
                }
                for x in kb.iter_mut() {
                    *x = f16t(gauss());
                }
                for x in vb.iter_mut() {
                    *x = f16t(gauss());
                }
            }
            let q = DeviceBuffer::alloc(devid, n * b * 2).unwrap();
            let k = DeviceBuffer::alloc(devid, nk * b * 2).unwrap();
            let v = DeviceBuffer::alloc(devid, nk * b * 2).unwrap();
            let o = DeviceBuffer::alloc(devid, n * b * 2).unwrap();
            // Kernel writes 128 rounded rows per (batch, head) — same sizing
            // as the dense test (b * QH * ceil(seq/128) * 128 * 4).
            let lse =
                DeviceBuffer::alloc(devid, b * heads as usize * (seq as usize).div_ceil(128) * 128 * 4).unwrap();
            copy(&mut MemRef::Device(&q), &mut MemRef::Host(&hq), n * b * 2, None).unwrap();
            copy(&mut MemRef::Device(&k), &mut MemRef::Host(&hk), nk * b * 2, None).unwrap();
            copy(&mut MemRef::Device(&v), &mut MemRef::Host(&hv), nk * b * 2, None).unwrap();
            let run = |vi: usize, seqlen: u32| -> (f32, f32) {
                fmha.launch_batched_prefill_variant(
                    dev,
                    vi,
                    q.as_ptr() as *const u16,
                    k.as_ptr() as *const u16,
                    v.as_ptr() as *const u16,
                    o.as_ptr() as *mut u16,
                    lse.as_ptr() as *mut f32,
                    seqlen,
                    batch,
                    heads,
                    kv_heads,
                    d,
                )
                .unwrap();
                stream.synchronize().unwrap();
                let mut h = HostBuffer::alloc(n * b * 2).unwrap();
                copy(&mut MemRef::Host(&mut h), &mut MemRef::Device(&o), n * b * 2, None).unwrap();
                let bits = unsafe { std::slice::from_raw_parts(h.as_ptr() as *const u16, n * b) };
                let lse_n = b * heads as usize * (seq as usize).div_ceil(128) * 128;
                let mut hlse = HostBuffer::alloc(lse_n * 4).unwrap();
                copy(
                    &mut MemRef::Host(&mut hlse),
                    &mut MemRef::Device(&lse),
                    lse_n * 4,
                    None,
                )
                .unwrap();
                let lsef: &[f32] =
                    unsafe { std::slice::from_raw_parts(hlse.as_ptr() as *const f32, lse_n) };
                // Row 64 of batch 0: element 64*(b*nqk) (row stride is
                // b*nqk — 64*nqk alone is row 32 at batch=2, a probe bug
                // that made (100,2) look cured while the gate failed).
                (f16_to_f32(bits[64 * b * nqk]), lsef[64])
            };
            let (v2_64, lse64) = run(2, seq);
            let (v0_64, _) = run(0, seq);
            let (v2b_64, _) = run(2, seq);
            println!(
                "odd-mn (100,batch={batch}): v2 O[64]={v2_64:.4e} LSE[64]={lse64:.4e} \
                 then v0 O[64]={v0_64:.4e} then v2-again O[64]={v2b_64:.4e}"
            );
        }
    }

    /// Full-gate-sequence replica (S1-7): the dense gate's (100,2) fails
    /// deterministically (O row 64 zero) while every isolated probe of the
    /// same shape passes. The one remaining difference is context history:
    /// the gate runs SIX prior shapes, each with its own fresh stream +
    /// fresh FmhaKernels (constructor warmup) + pick launch, before
    /// (100,2). Replicate the exact case loop and check whether the last
    /// shape reproduces the failure.
    #[test]
    #[ignore]
    fn fmha_gate_sequence_replica() {
        let Some((_ctx, dev, devid)) = setup() else { return };
        let arch = reinfer_cuda::arch::resolve_arch().unwrap();
        let cache = std::env::temp_dir().join("reinfer-jit-fmha-prefill");
        let (heads, kv_heads) = (8u32, 4u32);
        let d = 128u32;
        let nqk = (heads * d) as usize;
        let kvk = (kv_heads * d) as usize;
        for (seq, batch) in [
            (256u32, 1u32),
            (256, 3),
            (1024, 1),
            (1024, 3),
            (4096, 1),
            (4096, 3),
            (100, 2),
        ]
        {
            let stream = CudaStream::new(devid).unwrap();
            let fmha = FmhaKernels::new(&arch, Some(cache.clone()), stream.clone()).unwrap();
            let b = batch as usize;
            let n = seq as usize * nqk;
            let nk = seq as usize * kvk;
            let (hq, hk, hv) = gate_draw(seq, batch);
            let q = DeviceBuffer::alloc(devid, n * b * 2).unwrap();
            let k = DeviceBuffer::alloc(devid, nk * b * 2).unwrap();
            let v = DeviceBuffer::alloc(devid, nk * b * 2).unwrap();
            let o = DeviceBuffer::alloc(devid, n * b * 2).unwrap();
            let lse = DeviceBuffer::alloc(
                devid,
                b * heads as usize * (seq as usize).div_ceil(128) * 128 * 4,
            )
            .unwrap();
            copy(&mut MemRef::Device(&q), &mut MemRef::Host(&hq), n * b * 2, None).unwrap();
            copy(&mut MemRef::Device(&k), &mut MemRef::Host(&hk), nk * b * 2, None).unwrap();
            copy(&mut MemRef::Device(&v), &mut MemRef::Host(&hv), nk * b * 2, None).unwrap();
            fmha.launch_batched_prefill(
                dev,
                q.as_ptr() as *const u16,
                k.as_ptr() as *const u16,
                v.as_ptr() as *const u16,
                o.as_ptr() as *mut u16,
                lse.as_ptr() as *mut f32,
                seq,
                batch,
                heads,
                kv_heads,
                d,
            )
            .unwrap();
            fmha.sync_stream().unwrap();
            if seq == 100 {
                // Last shape: check the odd-MN second sub-block (row 64),
                // then run a same-buffer v0 to get the proven-correct value.
                let mut ho = HostBuffer::alloc(n * b * 2).unwrap();
                copy(&mut MemRef::Host(&mut ho), &mut MemRef::Device(&o), n * b * 2, None).unwrap();
                let ob =
                    unsafe { std::slice::from_raw_parts(ho.as_ptr() as *const u16, n * b) };
                let lse_n = b * heads as usize * (seq as usize).div_ceil(128) * 128;
                let mut hlse = HostBuffer::alloc(lse_n * 4).unwrap();
                copy(
                    &mut MemRef::Host(&mut hlse),
                    &mut MemRef::Device(&lse),
                    lse_n * 4,
                    None,
                )
                .unwrap();
                let lsef: &[f32] =
                    unsafe { std::slice::from_raw_parts(hlse.as_ptr() as *const f32, lse_n) };
                println!(
                    "replica (100,2): pick-v2 O[64]={} LSE[64]={}",
                    f16_to_f32(ob[64 * b * nqk]),
                    lsef[64]
                );
                fmha.launch_batched_prefill_variant(
                    dev,
                    0,
                    q.as_ptr() as *const u16,
                    k.as_ptr() as *const u16,
                    v.as_ptr() as *const u16,
                    o.as_ptr() as *mut u16,
                    lse.as_ptr() as *mut f32,
                    seq,
                    batch,
                    heads,
                    kv_heads,
                    d,
                )
                .unwrap();
                fmha.sync_stream().unwrap();
                copy(&mut MemRef::Host(&mut ho), &mut MemRef::Device(&o), n * b * 2, None).unwrap();
                let ob2 =
                    unsafe { std::slice::from_raw_parts(ho.as_ptr() as *const u16, n * b) };
                println!(
                    "replica (100,2): then v0 O[64]={}",
                    f16_to_f32(ob2[64 * b * nqk])
                );
            }
        }
    }

    /// Large-grid odd-MN probe (S1-7): the odd-MN first-launch race is
    /// proven at grid.x=1 (100-token shapes). The engine's real prompts are
    /// 256 (even), 2047 and 2659 (odd, grid.x=16/21) — if the race also
    /// hits large grid.x, the LAST CTA's second sub-block (rows 1984.. for
    /// 2047) would be zero/stale on a geometry change after a 256 prefill
    /// (the preflight fires once per instance; a later odd shape is the
    /// first launch of ITS geometry). Compare pick-v2@2047 vs v0@2047 on
    /// the same buffers, rows 1920-2047.
    #[test]
    #[ignore]
    fn fmha_v2_odd_large_grid_probe() {
        let Some((_ctx, dev, devid)) = setup() else { return };
        let arch = reinfer_cuda::arch::resolve_arch().unwrap();
        let cache = std::env::temp_dir().join("reinfer-jit-fmha-prefill");
        let (heads, kv_heads) = (8u32, 4u32);
        let d = 128u32;
        let nqk = (heads * d) as usize;
        let kvk = (kv_heads * d) as usize;
        let stream = CudaStream::new(devid).unwrap();
        let fmha = FmhaKernels::new(&arch, Some(cache), stream.clone()).unwrap();
        let run = |seq: u32, o: &DeviceBuffer, q: &DeviceBuffer, k: &DeviceBuffer, v: &DeviceBuffer, lse: &DeviceBuffer| {
            fmha.launch_batched_prefill(
                dev,
                q.as_ptr() as *const u16,
                k.as_ptr() as *const u16,
                v.as_ptr() as *const u16,
                o.as_ptr() as *mut u16,
                lse.as_ptr() as *mut f32,
                seq,
                1,
                heads,
                kv_heads,
                d,
            )
            .unwrap();
            stream.synchronize().unwrap();
        };
        // (256,1) first: fires the instance preflight; the engine's real
        // flow always does a 256-token prefill before longer prompts.
        let (hq, hk, hv) = gate_draw(256, 1);
        let (q, k, v, o, lse) = alloc_seq(devid, 256, nqk, kvk, heads, &hq, &hk, &hv);
        run(256, &o, &q, &k, &v, &lse);
        // 2047 (odd, grid.x=16): NO preflight (second launch of the
        // instance, first launch of this geometry).
        let seq = 2047u32;
        let (hq, hk, hv) = gate_draw(seq, 1);
        let (q, k, v, o, lse) = alloc_seq(devid, seq, nqk, kvk, heads, &hq, &hk, &hv);
        run(seq, &o, &q, &k, &v, &lse);
        // Rows 1920..2047 of batch 0 (the last CTA's second sub-block
        // starts at 1984).
        let n = seq as usize * nqk;
        let mut ho = HostBuffer::alloc(n * 2).unwrap();
        copy(&mut MemRef::Host(&mut ho), &mut MemRef::Device(&o), n * 2, None).unwrap();
        let bits = unsafe { std::slice::from_raw_parts(ho.as_ptr() as *const u16, n) };
        let mut v2 = vec![0u16; (seq as usize - 1920) * nqk];
        v2.copy_from_slice(&bits[1920 * nqk..seq as usize * nqk]);
        // Same-buffer v0 reference for the same rows.
        run(seq, &o, &q, &k, &v, &lse);
        copy(&mut MemRef::Host(&mut ho), &mut MemRef::Device(&o), n * 2, None).unwrap();
        let bits0 = unsafe { std::slice::from_raw_parts(ho.as_ptr() as *const u16, n) };
        let mut worst = 0usize;
        let mut worst_row = 0usize;
        for (r, (a, b)) in v2.iter().zip(&bits0[1920 * nqk..seq as usize * nqk]).enumerate() {
            if a != b {
                worst = worst.max((a ^ b).count_ones() as usize);
                worst_row = r;
            }
        }
        // Element 0 of row 2046 (the last valid row; 2047 is the block
        // tail) and the first-affected row's element.
        println!(
            "large-odd 2047: v2@2047 vs v0@2047 rows 1920-2047 max bit diff={worst} \
             (0 = identical) at row-offset {worst_row}; O[2046,0]: v2={} v0={}",
            f16_to_f32(bits[2046 * nqk]),
            f16_to_f32(bits0[2046 * nqk])
        );
    }

    /// Allocate q/k/v/o/lse and upload (probe helper for the odd-MN tests).
    fn alloc_seq(
        devid: DeviceId,
        seq: u32,
        nqk: usize,
        kvk: usize,
        heads: u32,
        hq: &HostBuffer,
        hk: &HostBuffer,
        hv: &HostBuffer,
    ) -> (DeviceBuffer, DeviceBuffer, DeviceBuffer, DeviceBuffer, DeviceBuffer) {
        let n = seq as usize * nqk;
        let nk = seq as usize * kvk;
        let q = DeviceBuffer::alloc(devid, n * 2).unwrap();
        let k = DeviceBuffer::alloc(devid, nk * 2).unwrap();
        let v = DeviceBuffer::alloc(devid, nk * 2).unwrap();
        let o = DeviceBuffer::alloc(devid, n * 2).unwrap();
        let lse = DeviceBuffer::alloc(
            devid,
            heads as usize * (seq as usize).div_ceil(128) * 128 * 4,
        )
        .unwrap();
        copy(&mut MemRef::Device(&q), &mut MemRef::Host(hq), n * 2, None).unwrap();
        copy(&mut MemRef::Device(&k), &mut MemRef::Host(hk), nk * 2, None).unwrap();
        copy(&mut MemRef::Device(&v), &mut MemRef::Host(hv), nk * 2, None).unwrap();
        (q, k, v, o, lse)
    }

    // ============ S1-7: FMHA-leg vs referee differential ============
    //
    // Parity p5 (s=20) overflows the f16 channel at layer 27's FFN and the
    // engine A/B test (fmha-leg vs dense step-loop at 256) drifts 2.4e1 vs
    // a 2.3e-1 gate. Which leg is wrong? This probe runs every prefill leg
    // (FMHA-f16, FMHA-separated-QKV, dense-f16 step loop, f32 channel)
    // against the llama.cpp referee (golden) on the p5 prompt and a
    // ~256-token prompt. Diagnostic only — no hard asserts (the referee
    // protocol is parity.rs's; REINFER_REFEREE / REINFER_REFEREE_GGUF envs).

    /// Engine load with the f32-channel env set/unset around the call
    /// (Engine::load reads REINFER_PARITY_F32 once at load).
    fn load_engine_leg(
        devid: DeviceId,
        arch: &str,
        cache: &PathBuf,
        model: &PathBuf,
        parity_f32: bool,
    ) -> Engine {
        // SAFETY: test-only env mutation on the main thread, no concurrent
        // readers of REINFER_PARITY_F32 within this process.
        if parity_f32 {
            unsafe { std::env::set_var("REINFER_PARITY_F32", "on") };
        } else {
            unsafe { std::env::remove_var("REINFER_PARITY_F32") };
        }
        let e = Engine::load(devid, arch, Some(cache.clone()), model, 4096).expect("engine load");
        unsafe { std::env::remove_var("REINFER_PARITY_F32") };
        e
    }

    fn load_tokenizer(model: &PathBuf) -> Tokenizer {
        let tok: serde_json::Value = serde_json::from_slice(
            &std::fs::read(model.join("tokenizer.json")).expect("tokenizer.json"),
        )
        .expect("tokenizer json");
        let cfg: serde_json::Value = serde_json::from_slice(
            &std::fs::read(model.join("tokenizer_config.json")).expect("tokenizer_config.json"),
        )
        .expect("tokcfg json");
        Tokenizer::from_hf_json(&tok, &cfg).expect("hf tokenizer")
    }

    /// llama-referee prefill-end logits for `prompt` (magic/parse per the
    /// parity.rs protocol).
    fn referee_prefill_logits(prompt: &str, n_steps: usize) -> (Vec<u32>, Vec<f32>) {
        let bin = std::env::var("REINFER_REFEREE").expect("REINFER_REFEREE");
        let gguf = std::env::var("REINFER_REFEREE_GGUF").expect("REINFER_REFEREE_GGUF");
        let threads = std::env::var("REINFER_REFEREE_THREADS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8);
        let out =
            std::env::temp_dir().join(format!("reinfer-fmha-referee-{}.bin", std::process::id()));
        let out_s = out.to_string_lossy().into_owned();
        let mut child = Command::new(&bin)
            .args([
                "-m",
                &gguf,
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
            .unwrap_or_else(|e| panic!("spawn referee {bin}: {e}"));
        child.stdin.take().unwrap().write_all(prompt.as_bytes()).unwrap();
        let cap = child.wait_with_output().expect("referee wait");
        let err = String::from_utf8_lossy(&cap.stderr);
        assert!(cap.status.success(), "referee failed: {err}");
        let data = std::fs::read(&out).unwrap_or_else(|e| panic!("read {out_s}: {e}"));
        let _ = std::fs::remove_file(&out);
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
        let n_steps_r = rd(&mut off) as usize;
        assert_eq!(n_steps_r, n_steps, "referee step count");
        let mut step0: Option<Vec<f32>> = None;
        for _ in 0..n_steps {
            let _token = rd(&mut off);
            let mut logits = Vec::with_capacity(n_vocab);
            for _ in 0..n_vocab {
                let b: [u8; 4] = data[off..off + 4].try_into().unwrap();
                off += 4;
                logits.push(f32::from_le_bytes(b));
            }
            if step0.is_none() {
                step0 = Some(logits);
            }
        }
        assert_eq!(off, data.len(), "referee trailing bytes");
        (prompt_ids, step0.unwrap())
    }

    /// Dense f16 step-loop prefill row (the fmha=false A/B leg).
    fn dense_f16_row(eng: &mut Engine, ids: &[u32]) -> Vec<f32> {
        let mut lg = Vec::new();
        for (i, &t) in ids.iter().enumerate() {
            lg = eng.step(t, i, i + 1).expect("dense step");
        }
        lg
    }

    fn topk_str(lg: &[f32], n: usize) -> String {
        let mut idx: Vec<usize> = (0..lg.len()).collect();
        idx.sort_by(|&i, &j| lg[j].total_cmp(&lg[i]));
        idx[..n.min(idx.len())]
            .iter()
            .map(|&i| format!("{i}:{:.2}", lg[i]))
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// A text tokenizing to at least `target` tokens (repeated pangram).
    fn build_text_of_len(tok: &Tokenizer, target: usize) -> (String, Vec<u32>) {
        const SENT: &str = "The quick brown fox jumps over the lazy dog. ";
        let mut text = String::new();
        while tok.encode(&text, false).expect("encode").len() < target {
            text.push_str(SENT);
        }
        let ids = tok.encode(&text, false).expect("encode");
        (text, ids)
    }

    /// Diagnostic differential on p5 (s=20) and a ~256-token prompt.
    #[test]
    #[ignore]
    fn fmha_leg_vs_referee_differential() {
        let Some((_ctx, _dev, devid)) = setup() else { return };
        let Some(model_dir) = std::env::var_os("REINFER_MODEL_DIR").map(PathBuf::from) else {
            eprintln!("fmha_leg_vs_referee: REINFER_MODEL_DIR unset (skip)");
            return;
        };
        if std::env::var_os("REINFER_REFEREE").is_none() {
            eprintln!("fmha_leg_vs_referee: REINFER_REFEREE unset (skip)");
            return;
        }
        let arch = reinfer_cuda::arch::resolve_arch().expect("arch");
        let cache = std::env::temp_dir().join("reinfer-jit-fmha-referee");
        let tok = load_tokenizer(&model_dir);

        let p5 = "9.11 和 9.9 哪个更大？请回答并解释。";
        let (text256, ids256) = build_text_of_len(&tok, 256);

        let report = |label: &str, ids: &[u32], row: &Vec<f32>, ref_lg: &[f32]| {
            let finite = row.iter().all(|l| l.is_finite());
            let drift = if finite {
                let rowmax = row
                    .iter()
                    .chain(ref_lg.iter())
                    .map(|v| v.abs())
                    .fold(0.0f32, f32::max)
                    .max(1e-12);
                row.iter()
                    .zip(ref_lg.iter())
                    .map(|(&a, &b)| (a - b).abs() / rowmax)
                    .fold(0.0f32, f32::max)
            } else {
                f32::NAN
            };
            println!(
                "  {label} s={:4}: finite={finite:<5} rel-drift-vs-referee={drift:.3e} top5 [{}]",
                ids.len(),
                topk_str(row, 5)
            );
        };

        // --- p5 (s=20) ---
        let ids5 = tok.encode(p5, false).expect("encode p5");
        println!("fmha_leg_vs_referee: p5 s={} (parity expects 20)", ids5.len());
        let mut eng_f16 = load_engine_leg(devid, &arch, &cache, &model_dir, false);
        let row_f5 = eng_f16.prefill_batch(&ids5).expect("fmha prefill p5");
        // SAFETY: test-only env mutation on the main thread (single-threaded test).
        unsafe { std::env::set_var("REINFER_PREFILL_SEP_QKV", "1") };
        let row_s5 = eng_f16.prefill_batch(&ids5).expect("fmha sep-qkv prefill p5");
        unsafe { std::env::remove_var("REINFER_PREFILL_SEP_QKV") };
        let mut eng_d5 = load_engine_leg(devid, &arch, &cache, &model_dir, false);
        let row_d5 = dense_f16_row(&mut eng_d5, &ids5);
        let mut eng_32 = load_engine_leg(devid, &arch, &cache, &model_dir, true);
        let row_32_5 = eng_32.prefill_batch(&ids5).expect("f32 prefill p5");
        let (ref_ids5, ref_lg5) = referee_prefill_logits(p5, 1);
        assert_eq!(ref_ids5, ids5, "tier ① p5");
        report("fmha-f16   ", &ids5, &row_f5, &ref_lg5);
        report("fmha-sep-qkv", &ids5, &row_s5, &ref_lg5);
        report("dense-f16  ", &ids5, &row_d5, &ref_lg5);
        report("f32-channel", &ids5, &row_32_5, &ref_lg5);

        // --- ~256 tokens ---
        println!(
            "fmha_leg_vs_referee: s256 text len = {} (target 256)",
            ids256.len()
        );
        let row_f256 = eng_f16.prefill_batch(&ids256).expect("fmha prefill 256");
        let mut eng_d256 = load_engine_leg(devid, &arch, &cache, &model_dir, false);
        let row_d256 = dense_f16_row(&mut eng_d256, &ids256);
        let mut eng_32b = load_engine_leg(devid, &arch, &cache, &model_dir, true);
        let row_32_256 = eng_32b.prefill_batch(&ids256).expect("f32 prefill 256");
        let (ref_ids256, ref_lg256) = referee_prefill_logits(&text256, 1);
        assert_eq!(ref_ids256, ids256, "tier ① s256");
        report("fmha-f16   ", &ids256, &row_f256, &ref_lg256);
        report("dense-f16  ", &ids256, &row_d256, &ref_lg256);
        report("f32-channel", &ids256, &row_32_256, &ref_lg256);
    }

    /// Isolated fused-QKV GEMM layout probe: which fused [h x N] weight
    /// construction reproduces the three separated q/k/v GEMMs (ground
    /// truth) under gemm1r's exact cublas call shape (OP_N/OP_N, 32F
    /// compute, amat ld = n, bmat ld = k, cmat ld = n —
    /// `GemmPlan::col_major_swap_f16` with a = weight, b = xn)? Candidates:
    /// A = byte stack (q16 ++ k16 ++ v16 — the current engine construction),
    /// B = per-row column join (fused row k = q row k ++ k row k ++ v row k),
    /// C = transposed per-row join (control). The D7 failure mode (99.97%
    /// of elements over the gate for BOTH constructions) means the fused
    /// weight layout is NOT the whole story — this probe isolates the
    /// cublas level before any kernel/cast/downstream code is involved.
    /// Diagnostic only — no hard asserts.
    #[test]
    #[ignore]
    fn fused_qkv_gemm_layout_probe() {
        let Some((_ctx, dev, devid)) = setup() else { return };
        let Some(model_dir) = std::env::var_os("REINFER_MODEL_DIR").map(PathBuf::from) else {
            eprintln!("fused_qkv_probe: REINFER_MODEL_DIR unset (skip)");
            return;
        };
        let cfg: serde_json::Value = serde_json::from_slice(
            &std::fs::read(model_dir.join("config.json")).expect("config.json"),
        )
        .expect("config json");
        let h = cfg["hidden_size"].as_u64().expect("hidden_size") as usize;
        let qh = cfg["num_attention_heads"].as_u64().expect("q_heads") as usize;
        let kh = cfg["num_key_value_heads"].as_u64().expect("kv_heads") as usize;
        let d = cfg["head_dim"].as_u64().expect("head_dim") as usize;
        let nqk = qh * d;
        let kvk = kh * d;
        let n = nqk + 2 * kvk;
        let s = 256usize;
        eprintln!("fused_qkv_probe: h={h} nqk={nqk} kvk={kvk} N={n} s={s}");

        // --- layer-0 q/k/v raw tensor bytes (seek; no full-file read) ---
        use std::io::{Read, Seek, SeekFrom};
        let mut f = std::fs::File::open(model_dir.join("model.safetensors")).expect("open");
        let mut hdr_b = [0u8; 8];
        f.read_exact(&mut hdr_b).unwrap();
        let hdr_len = u64::from_le_bytes(hdr_b) as usize;
        let mut hdr = vec![0u8; hdr_len];
        f.read_exact(&mut hdr).unwrap();
        let header: serde_json::Value = serde_json::from_slice(&hdr).expect("header json");
        let data_start = 8 + hdr_len;
        let raw = |f: &mut std::fs::File, name: &str| -> Vec<u8> {
            let t = &header[name];
            let o = t["data_offsets"].as_array().expect("offsets");
            let a = o[0].as_u64().unwrap() as usize;
            let b = o[1].as_u64().unwrap() as usize;
            f.seek(SeekFrom::Start((data_start + a) as u64)).unwrap();
            let mut buf = vec![0u8; b - a];
            f.read_exact(&mut buf).unwrap();
            buf
        };
        // Replica of engine.rs `to_f16_rm` (bit-exact byte placement).
        let to_f16_rm = |tbytes: &[u8], out: usize, inp: usize| -> Vec<u8> {
            let mut w16 = vec![0u8; out * inp * 2];
            for r in 0..out {
                for c in 0..inp {
                    let src = (r * inp + c) * 2;
                    w16[c * out * 2 + r * 2] = tbytes[src];
                    w16[c * out * 2 + r * 2 + 1] = tbytes[src + 1];
                }
            }
            w16
        };
        let q16 = to_f16_rm(&raw(&mut f, "model.layers.0.self_attn.q_proj.weight"), nqk, h);
        let k16 = to_f16_rm(&raw(&mut f, "model.layers.0.self_attn.k_proj.weight"), kvk, h);
        let v16 = to_f16_rm(&raw(&mut f, "model.layers.0.self_attn.v_proj.weight"), kvk, h);
        // A: byte stack (current engine construction).
        let mut stack = q16.clone();
        stack.extend_from_slice(&k16);
        stack.extend_from_slice(&v16);
        // B: per-row column join (fused row k = q row k ++ k row k ++ v row k).
        let (nqb, nkb, nb) = (nqk * 2, kvk * 2, n * 2);
        let mut join = vec![0u8; nb * h];
        for r in 0..h {
            let row = r * nb;
            join[row..row + nqb].copy_from_slice(&q16[r * nqb..(r + 1) * nqb]);
            join[row + nqb..row + nqb + nkb].copy_from_slice(&k16[r * nkb..(r + 1) * nkb]);
            join[row + nqb + nkb..(r + 1) * nb].copy_from_slice(&v16[r * nkb..(r + 1) * nkb]);
        }
        // --- uploads / gemm1r-shaped runs ---
        let stream = CudaStream::new(devid).unwrap();
        let gemm = reinfer_cuda::gemm::Gemm::new(dev).unwrap();
        let mut xnh = vec![0u8; s * h * 2];
        {
            let mut lcg = Lcg(0x1234_5678_9abc_def0);
            for ch in xnh.chunks_exact_mut(2) {
                let v: u16 = 0x3800 | ((lcg.next_u64() >> 42) as u16 & 0x3ff);
                ch.copy_from_slice(&v.to_le_bytes());
            }
        }
        let hb_xn = HostBuffer::alloc(xnh.len()).unwrap();
        unsafe { std::ptr::copy_nonoverlapping(xnh.as_ptr(), hb_xn.as_ptr() as *mut u8, xnh.len()) };
        let xn = DeviceBuffer::alloc(devid, xnh.len()).unwrap();
        copy(&mut MemRef::Device(&xn), &MemRef::Host(&hb_xn), xnh.len(), None).unwrap();
        let run = |w: &[u8], wn: usize| -> Vec<f32> {
            assert_eq!(w.len(), wn * h * 2);
            let hw = HostBuffer::alloc(w.len()).unwrap();
            unsafe { std::ptr::copy_nonoverlapping(w.as_ptr(), hw.as_ptr() as *mut u8, w.len()) };
            let wd = DeviceBuffer::alloc(devid, w.len()).unwrap();
            copy(&mut MemRef::Device(&wd), &MemRef::Host(&hw), w.len(), None).unwrap();
            let cb = DeviceBuffer::alloc(devid, s * wn * 4).unwrap();
            // NOTE: col_major_swap_f16's first param is the row-major A
            // (xn [s x h]), second is the row-major B (weight [h x wn]) —
            // the constructor swaps them into cublas A/B (gemm1r shape).
            let plan = reinfer_cuda::gemm::GemmPlan::col_major_swap_f16(
                xn.as_ptr() as *const u16,
                wd.as_ptr() as *const u16,
                cb.as_ptr() as *mut f32,
                s,
                wn,
                h,
            );
            gemm.execute(&stream, &plan).unwrap();
            let hc = HostBuffer::alloc(s * wn * 4).unwrap();
            copy(&mut MemRef::Host(&hc), &MemRef::Device(&cb), s * wn * 4, None).unwrap();
            let cf = unsafe { std::slice::from_raw_parts(hc.as_ptr() as *const f32, s * wn) };
            cf.to_vec()
        };
        let cq = run(&q16, nqk);
        let ck = run(&k16, kvk);
        let cv = run(&v16, kvk);
        let ca = run(&stack, n);
        let cb = run(&join, n);

        // --- host-side reference: validates the separated leg (probe itself) ---
        let f16v = |b: &[u8]| -> f32 {
            let bits = u16::from_le_bytes([b[0], b[1]]);
            let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
            let exp = (bits >> 10) & 0x1f;
            let man = bits & 0x3ff;
            if exp == 0 {
                sign * (man as f32) * 2f32.powi(-24)
            } else {
                sign * (1.0 + man as f32 / 1024.0) * 2f32.powi(exp as i32 - 15)
            }
        };
        let mut ref_worst = 0.0f32;
        for j in 0..4 {
            for i in 0..4 {
                let mut acc = 0.0f32;
                for k in 0..h {
                    acc += f16v(&xnh[(j * h + k) * 2..(j * h + k) * 2 + 2])
                        * f16v(&q16[(k * nqk + i) * 2..(k * nqk + i) * 2 + 2]);
                }
                ref_worst = ref_worst.max((acc - cq[j * nqk + i]).abs());
            }
        }
        eprintln!(
            "fused_qkv_probe: host-ref vs separated q: worst |d| = {ref_worst:.3e} (probe sanity)"
        );

        // --- block comparisons (D7 gate formula: |a-b| <= 3e-2 + 1e-2*max) ---
        // `off` = fused column offset of the compared block (0 / nqk /
        // nqk+kvk); the separated leg's buffer is its own [s x ncol] rows.
        let cmp = |label: &str, fused: &[f32], off: usize, sep: &[f32]| {
            let ncol = sep.len() / s;
            let mut worst = 0.0f32;
            let mut over = 0usize;
            for j in 0..s {
                for i in 0..ncol {
                    let a = fused[j * n + off + i];
                    let b = sep[j * ncol + i];
                    let d = (a - b).abs();
                    let th = 3e-2 + 1e-2 * a.abs().max(b.abs());
                    worst = worst.max(d);
                    if d > th {
                        over += 1;
                    }
                }
            }
            eprintln!(
                "fused_qkv_probe: {label}: max|a-b|={worst:.3e} over_d7={over}/{}",
                s * ncol
            );
        };
        cmp("A-stack/q", &ca, 0, &cq);
        cmp("A-stack/k", &ca, nqk, &ck);
        cmp("A-stack/v", &ca, nqk + kvk, &cv);
        cmp("B-join/q", &cb, 0, &cq);
        cmp("B-join/k", &cb, nqk, &ck);
        cmp("B-join/v", &cb, nqk + kvk, &cv);
    }

    /// S1-7 prefill microbench: fused-QKV vs separated-QKV prefill wall
    /// time per prompt length, same deterministic ids (the fused path is
    /// bit-identical, so the numbers are pure launch/GEMM/cast cost).
    /// REINFER_PREFILL_PROFILE=1 additionally prints the per-kernel GPU
    /// attribution from the engine profiler. Diagnostic — no asserts.
    #[test]
    fn prefill_qkv_leg_microbench() {
        let Some((_ctx, _dev, devid)) = setup() else { return };
        let Some(model_dir) = std::env::var_os("REINFER_MODEL_DIR").map(PathBuf::from) else {
            eprintln!("prefill_qkv_leg_microbench: REINFER_MODEL_DIR unset (skip)");
            return;
        };
        let arch = reinfer_cuda::arch::resolve_arch().expect("arch");
        let cache = std::env::temp_dir().join("reinfer-jit-fmha-referee");
        let mut eng = Engine::load(devid, &arch, Some(cache), &model_dir, 8192).expect("engine load");
        // Warmup: pays the first-launch JIT compile / preflight / allocator
        // cost so the timed fused seq=256 run is not polluted.
        let warm: Vec<u32> = (0..64).map(|i| (i as u32) % 4096 + 3).collect();
        eng.prefill_batch(&warm).expect("warmup prefill");
        for seq in [256usize, 2047, 2659] {
            let ids: Vec<u32> = (0..seq).map(|i| (i as u32) % 4096 + 3).collect();
            for rep in 0..2 {
                for (label, sep) in [("fused", false), ("sep", true)] {
                    if sep {
                        unsafe { std::env::set_var("REINFER_PREFILL_SEP_QKV", "1") };
                    } else {
                        unsafe { std::env::remove_var("REINFER_PREFILL_SEP_QKV") };
                    }
                    let t0 = std::time::Instant::now();
                    eng.prefill_batch(&ids).expect("prefill");
                    let dt = t0.elapsed().as_secs_f64();
                    eprintln!(
                        "prefill seq={seq} rep={rep} {label}: {:.2} ms ({:.0} tok/s)",
                        dt * 1e3,
                        seq as f64 / dt
                    );
                }
            }
        }
        unsafe { std::env::remove_var("REINFER_PREFILL_SEP_QKV") };
    }

    /// S1-7 engine-level determinism check (v1-pick verification): the same
    /// engine + same prompt twice must be bit-identical at EVERY position.
    /// Under the old v2 pick the preflight fired only on layer 1 (one
    /// (shape, address) key for all 28 layers), so layers 2..28 ran the v2
    /// bare and every O row >= 128 held the PREVIOUS layer's values — the
    /// batch-leg stale chain. With the v1 pick (clean stores) + the v0
    /// preflight (bit-identical math) both runs must agree exactly.
    #[test]
    fn engine_prefill_determinism_v1() {
        let Some((_ctx, _dev, devid)) = setup() else { return };
        let Some(model_dir) = std::env::var_os("REINFER_MODEL_DIR").map(PathBuf::from) else {
            eprintln!("engine_prefill_determinism_v1: REINFER_MODEL_DIR unset (skip)");
            return;
        };
        let arch = reinfer_cuda::arch::resolve_arch().expect("arch");
        let cache = std::env::temp_dir().join("reinfer-jit-fmha-referee");
        let mut eng = Engine::load(devid, &arch, Some(cache), &model_dir, 8192).expect("engine load");
        let ids: Vec<u32> = (0..512usize).map(|i| (i as u32) % 4096 + 3).collect();
        let a = eng.prefill_batch(&ids).expect("prefill run 1");
        let b = eng.prefill_batch(&ids).expect("prefill run 2");
        assert_eq!(a.len(), b.len());
        let mut first_diff = None;
        let mut nbad = 0usize;
        for (i, (x, y)) in a.iter().zip(&b).enumerate() {
            if x.to_bits() != y.to_bits() {
                nbad += 1;
                if first_diff.is_none() {
                    first_diff = Some((i, *x, *y));
                }
            }
        }
        println!(
            "engine 512-tok prefill x2: nbad={nbad}/{} first_diff={first_diff:?}",
            a.len()
        );
        assert_eq!(nbad, 0, "v1-pick prefill runs must be bit-identical");
    }

    /// S1-7 diagnostic: single-shot FMHA prefill vs per-token step loop on
    /// FRESH engines, plus a repeated-call state check. The three-way
    /// probes (step_loop_divergence_probe / S1-9's probe_threeway) show the
    /// FMHA leg diverging from the dense/CPU legs exactly at seqlen >= 65;
    /// this test decides whether that is (a) a real single-shot FMHA bug or
    /// (b) state carried across repeated prefill_batch calls on one engine
    /// (the probe pattern real usage never performs). Diagnostic — no asserts.
    #[test]
    fn fmha_single_shot_vs_step_probe() {
        let Some((_ctx, _dev, devid)) = setup() else { return };
        let Some(model_dir) = std::env::var_os("REINFER_MODEL_DIR").map(PathBuf::from) else {
            eprintln!("fmha_single_shot_vs_step_probe: REINFER_MODEL_DIR unset (skip)");
            return;
        };
        let arch = reinfer_cuda::arch::resolve_arch().expect("arch");
        let cache = std::env::temp_dir().join("reinfer-jit-fmha-referee");
        let mkids = |s: usize| -> Vec<u32> { (0..s).map(|i| (i as u32) % 4096 + 3).collect() };
        let drift = |a: &[f32], b: &[f32]| -> f32 {
            a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
        };
        for s in [64usize, 65, 128, 129, 256] {
            let ids = mkids(s);
            // Leg A: fresh engine, ONE prefill_batch (single-shot FMHA).
            let mut eng = Engine::load(devid, &arch, Some(cache.clone()), &model_dir, 8192)
                .expect("engine load");
            let a = eng.prefill_batch(&ids).expect("single-shot prefill");
            drop(eng);
            // Leg B: fresh engine, step loop (dense decode path).
            let mut eng = Engine::load(devid, &arch, Some(cache.clone()), &model_dir, 8192)
                .expect("engine load");
            let mut b = Vec::new();
            for (k, &t) in ids.iter().enumerate() {
                b = eng.step(t, k, k + 1).expect("step");
            }
            drop(eng);
            eprintln!("  single-shot s={s}: drift fmha-vs-step {:.3e}", drift(&a, &b));
            // Leg C: repeated calls at the same s on ONE engine — does the
            // 130th call differ from the 1st (state carry-over)?
            let mut eng = Engine::load(devid, &arch, Some(cache.clone()), &model_dir, 8192)
                .expect("engine load");
            let c1 = eng.prefill_batch(&ids).expect("call 1");
            let mut last = c1.clone();
            for _ in 1..130 {
                last = eng.prefill_batch(&ids).expect("repeated call");
            }
            eprintln!(
                "  repeated s={s}: call1-vs-call130 drift {:.3e}",
                drift(&c1, &last)
            );
        }
    }

    /// S1-7 diagnostic: localize the engine_prefill_batch_vs_step_loop
    /// drift. The D7 gate proves the FMHA prefill leg is bit-identical to
    /// the separated prefill, so any cross-leg drift at prefill end lives
    /// in the per-token step path (decode segment). Compares per-position
    /// prefill-end logits: FMHA `prefill_batch(ids[..=k])` vs `eng.step`
    /// at position k; prints the positions whose drift exceeds 1e-1 plus
    /// the top-5 argmax at the worst one. Diagnostic — no asserts.
    #[test]
    fn step_loop_divergence_probe() {
        let Some((_ctx, _dev, devid)) = setup() else { return };
        let Some(model_dir) = std::env::var_os("REINFER_MODEL_DIR").map(PathBuf::from) else {
            eprintln!("step_loop_divergence_probe: REINFER_MODEL_DIR unset (skip)");
            return;
        };
        let arch = reinfer_cuda::arch::resolve_arch().expect("arch");
        let cache = std::env::temp_dir().join("reinfer-jit-fmha-referee");
        let topk = |lg: &[f32]| {
            let mut idx: Vec<usize> = (0..lg.len()).collect();
            idx.sort_by(|&i, &j| lg[j].total_cmp(&lg[i]));
            idx[..5.min(idx.len())]
                .iter()
                .map(|&i| format!("{i}:{:.2}", lg[i]))
                .collect::<Vec<_>>()
                .join(" ")
        };
        for s in [64usize, 256] {
            let ids: Vec<u32> = (0..s).map(|i| (i as u32) % 4096 + 3).collect();
            // Dense leg: per-token steps, capture each position's logits.
            let mut eng = Engine::load(devid, &arch, Some(cache.clone()), &model_dir, 8192)
                .expect("engine load");
            let mut dense = Vec::with_capacity(s);
            for (k, &t) in ids.iter().enumerate() {
                dense.push(eng.step(t, k, k + 1).expect("step"));
            }
            drop(eng);
            // FMHA leg: fresh engine, prefill_batch on each prefix.
            let mut eng = Engine::load(devid, &arch, Some(cache.clone()), &model_dir, 8192)
                .expect("engine load");
            let mut fmha = Vec::with_capacity(s);
            for k in 0..s {
                fmha.push(eng.prefill_batch(&ids[..=k]).expect("prefill"));
            }
            let mut worst = (0.0f32, 0usize);
            let mut nbad = 0usize;
            for (k, (a, b)) in fmha.iter().zip(&dense).enumerate() {
                let d = a
                    .iter()
                    .zip(b)
                    .map(|(x, y)| (x - y).abs())
                    .fold(0.0f32, f32::max);
                if d > worst.0 {
                    worst = (d, k);
                }
                if d > 1e-1 {
                    if nbad < 4 {
                        eprintln!(
                            "  s={s} bad pos={k}: drift={d:.3e} fmha-top5 [{}] dense-top5 [{}]",
                            topk(a),
                            topk(b)
                        );
                    }
                    nbad += 1;
                }
            }
            eprintln!(
                "  s={s}: worst drift {:.3e} at pos {}; {} bad positions (>1e-1)",
                worst.0, worst.1, nbad
            );
        }
    }

    /// RNE f32 -> f16 (copy of fmha_prefill.rs helper; the host f16-P
    /// reference needs it for the P quantization — flash-attn rounds P to
    /// f16 with RNE).
    fn f32_to_f16_rne(f: f32) -> u16 {
        let bits = f.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exp = ((bits >> 23) & 0xff) as i32;
        let man = bits & 0x7f_ffff;
        if exp == 0xff {
            return sign | 0x7c00 | ((man >> 13) & 0x3ff) as u16;
        }
        if exp == 0 && man == 0 {
            return sign;
        }
        let half_exp = exp - 127 + 15;
        if half_exp <= 0 {
            if half_exp < -14 {
                return sign;
            }
            let shift = (1 - half_exp + 13) as u32;
            let m = man | 0x800_000;
            let round = m >> (shift - 1) & 1;
            let sticky = (m & ((1 << (shift - 1)) - 1)) != 0;
            let q = m >> shift;
            let mut r = q;
            if round == 1 && (sticky || q & 1 == 1) {
                r += 1;
            }
            let max_sub = 0x3ff;
            return sign | (if r > max_sub { max_sub } else { r }) as u16;
        }
        if half_exp >= 31 {
            return sign | 0x7c00;
        }
        let round = (man >> 12) & 1;
        let sticky = (man & 0xfff) != 0;
        let mut q = man >> 13;
        if round == 1 && (sticky || q & 1 == 1) {
            q += 1;
            if q == 0x400 {
                if half_exp + 1 >= 31 {
                    return sign | 0x7c00;
                }
                return sign | (((half_exp + 1) as u16) << 10);
            }
        }
        sign | ((half_exp as u16) << 10) | q as u16
    }

    /// DECISIVE kernel-level probe (S1-7 regression): the engine's prefill
    /// produces garbage logits at seqlen >= 65 while every (8,4) kernel
    /// probe passes — the engine runs ONE FMHA launch per layer (28 per
    /// prefill) at (16,8) heads. This test replays the engine's exact
    /// launch pattern at kernel level: DecodeKernels first (engine-load
    /// order), fresh FmhaKernels (preflight fires on launch #1), 28
    /// consecutive pick-path launches of the SAME buffers, host f16-P
    /// flash replica as the oracle (gate 1e-3, fmha_prefill.rs criterion)
    /// on launch #1 and launch #28, plus per-launch determinism vs
    /// launch #1. Both head configs (16,8)=engine and (8,4)=known-good
    /// control run the same matrix.
    /// Host f16-P reference: O bits + true LSE per (s, h) for the given
    /// f32 q/k/v views (q already includes any scaling). LSE layout in
    /// the kernel: (bidb*h + bidh)*seqlen_q_rounded + s.
    #[allow(clippy::too_many_arguments)]
    fn fmha_host_ref(
        q32: &[f32],
        k32: &[f32],
        v32: &[f32],
        seq: usize,
        heads: usize,
        kv_heads: usize,
        d: usize,
    ) -> (Vec<u16>, Vec<f32>) {
        let nqk = heads * d;
        let kvk = kv_heads * d;
        let sq_r = (seq.div_ceil(128) * 128) as usize;
        let ratio = heads / kv_heads;
        let mut reff = vec![0u16; seq * nqk];
        let mut reflse = vec![f32::NAN; heads * sq_r];
        for s in 0..seq {
            for h in 0..heads {
                let kh = h / ratio;
                let qr = &q32[s * nqk + h * d..s * nqk + (h + 1) * d];
                let mut scores = vec![0.0f32; s + 1];
                for j in 0..=s {
                    let kr = &k32[j * kvk + kh * d..j * kvk + (kh + 1) * d];
                    scores[j] = qr.iter().zip(kr).map(|(&a, &b)| a * b).sum();
                }
                let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                let mut p = vec![0u16; s + 1];
                for j in 0..=s {
                    let e = (scores[j] - mx).exp();
                    sum += e;
                    p[j] = f32_to_f16_rne(e);
                }
                let base = s * nqk + h * d;
                for i in 0..d {
                    let mut acc = 0.0f32;
                    for j in 0..=s {
                        acc += f16_to_f32(p[j]) * v32[j * kvk + kh * d + i];
                    }
                    reff[base + i] = if sum != 0.0 && sum == sum {
                        f32_to_f16_rne(acc / sum)
                    } else {
                        0
                    };
                }
                reflse[h * sq_r + s] = mx + sum.ln();
            }
        }
        (reff, reflse)
    }

    #[test]
    #[ignore]
    fn fmha_engine_config_kernel_probe() {
        let Some((_ctx, dev, devid)) = setup() else { return };
        let stream = CudaStream::new(devid).unwrap();
        let arch = reinfer_cuda::arch::resolve_arch().unwrap();
        let cache = std::env::temp_dir().join("reinfer-jit-fmha-prefill");
        reinfer_cuda::decode::DecodeKernels::new(&arch, Some(cache.clone()), stream.clone())
            .unwrap();
        let d = 128u32;
        // ONE shared FmhaKernels for ALL cells (engine scenario: the
        // engine caches one instance for its lifetime and serves varying
        // seqlen). With the per-geometry preflight fix every cell must
        // pass; before the fix only the first cell (whose geometry the
        // one-shot preflight happened to prime) passed.
        let fmha = FmhaKernels::new(&arch, Some(cache.clone()), stream.clone()).unwrap();
        // One cell = (heads, kv_heads, seq, draw-class). The dense gate's
        // draw (gate_draw) is the known-good control; dense_draw differs
        // only in the Q scale point (scale before vs after f16 truncation).
        let mut cells: Vec<(u32, u32, u32, &str, (HostBuffer, HostBuffer, HostBuffer))> = Vec::new();
        // Keep the (256,8,4) dense cell's device buffers alive past its
        // cell so the later gate-draw cell at the same geometry cannot
        // reuse those addresses (allocator reuse would collide the
        // preflight key and silently skip the gate cell's preflight).
        let mut keepalive: Vec<DeviceBuffer> = Vec::new();
        for &(heads, kv_heads) in &[(16u32, 8u32), (8u32, 4u32)] {
            for &seq in &[65u32, 128, 129, 256] {
                cells.push((heads, kv_heads, seq, "dense_draw", dense_draw(seq, heads, kv_heads)));
            }
        }
        // Known-good control: the exact dense-gate config/draw.
        for &(seq, batch) in &[(256u32, 1u32), (65u32, 1u32)] {
            let (hq, hk, hv) = gate_draw(seq, batch);
            // The dense gate scales q AFTER the f16 truncation (qs16 =
            // f32_to_f16(f16_to_f32(q16) * scale)) — replicate exactly.
            let scale = 1.0f32 / (d as f32).sqrt();
            let nq = (seq * 8 * d) as usize;
            let qb = unsafe { std::slice::from_raw_parts_mut(hq.as_ptr() as *mut u8 as *mut u16, nq) };
            for b in qb.iter_mut() {
                *b = f32_to_f16_rne(f16_to_f32(*b) * scale);
            }
            cells.push((8, 4, seq, "gate_draw", (hq, hk, hv)));
        }
        for (heads, kv_heads, seq, draw, (hq, hk, hv)) in cells {
            let mut keep_this_cell = draw == "dense_draw" && heads == 8 && kv_heads == 4 && seq == 256;
            let nqk = (heads * d) as usize;
            let kvk = (kv_heads * d) as usize;
            let n = seq as usize * nqk;
            let nk = seq as usize * kvk;
            let sq_r = (seq.div_ceil(128) * 128) as usize;
            let q = DeviceBuffer::alloc(devid, n * 2).unwrap();
            let k = DeviceBuffer::alloc(devid, nk * 2).unwrap();
            let v = DeviceBuffer::alloc(devid, nk * 2).unwrap();
            let o = DeviceBuffer::alloc(devid, n * 2).unwrap();
            let lse = DeviceBuffer::alloc(devid, heads as usize * sq_r * 4).unwrap();
            let cell_addr = || {
                (
                    q.as_ptr() as usize, k.as_ptr() as usize, v.as_ptr() as usize,
                    o.as_ptr() as usize, lse.as_ptr() as usize,
                )
            };
            copy(&mut MemRef::Device(&q), &mut MemRef::Host(&hq), n * 2, None).unwrap();
            copy(&mut MemRef::Device(&k), &mut MemRef::Host(&hk), nk * 2, None).unwrap();
            copy(&mut MemRef::Device(&v), &mut MemRef::Host(&hv), nk * 2, None).unwrap();
            let q32: Vec<f32> = unsafe {
                std::slice::from_raw_parts(hq.as_ptr() as *const u16, n)
                    .iter().map(|&b| f16_to_f32(b)).collect()
            };
            let k32: Vec<f32> = unsafe {
                std::slice::from_raw_parts(hk.as_ptr() as *const u16, nk)
                    .iter().map(|&b| f16_to_f32(b)).collect()
            };
            let v32: Vec<f32> = unsafe {
                std::slice::from_raw_parts(hv.as_ptr() as *const u16, nk)
                    .iter().map(|&b| f16_to_f32(b)).collect()
            };
            let ratio = heads / kv_heads;
            // Host f16-P reference + true LSE per (s, h). LSE layout in the
            // kernel: (bidb*h + bidh)*seqlen_q_rounded + s.
            let mut reff = vec![0u16; n];
            let mut reflse = vec![f32::NAN; heads as usize * sq_r];
            for s in 0..seq as usize {
                for h in 0..heads as usize {
                    let kh = h / ratio as usize;
                    let qr = &q32[s * nqk + h * d as usize..s * nqk + (h + 1) * d as usize];
                    let mut scores = vec![0.0f32; s + 1];
                    for j in 0..=s {
                        let kr = &k32[j * kvk + kh * d as usize..j * kvk + (kh + 1) * d as usize];
                        scores[j] = qr.iter().zip(kr).map(|(&a, &b)| a * b).sum();
                    }
                    let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0.0f32;
                    let mut p = vec![0u16; s + 1];
                    for j in 0..=s {
                        let e = (scores[j] - mx).exp();
                        sum += e;
                        p[j] = f32_to_f16_rne(e);
                    }
                    let base = s * nqk + h * d as usize;
                    for i in 0..d as usize {
                        let mut acc = 0.0f32;
                        for j in 0..=s {
                            acc += f16_to_f32(p[j]) * v32[j * kvk + kh * d as usize + i];
                        }
                        reff[base + i] = if sum != 0.0 && sum == sum {
                            f32_to_f16_rne(acc / sum)
                        } else {
                            0
                        };
                    }
                    reflse[h * sq_r + s] = mx + sum.ln();
                }
            }
            let launch = |fmha: &FmhaKernels| -> Vec<u16> {
                fmha.launch_batched_prefill(
                    dev,
                    q.as_ptr() as *const u16,
                    k.as_ptr() as *const u16,
                    v.as_ptr() as *const u16,
                    o.as_ptr() as *mut u16,
                    lse.as_ptr() as *mut f32,
                    seq,
                    1,
                    heads,
                    kv_heads,
                    d,
                )
                .unwrap();
                stream.synchronize().unwrap();
                let mut h = HostBuffer::alloc(n * 2).unwrap();
                copy(&mut MemRef::Host(&mut h), &mut MemRef::Device(&o), n * 2, None)
                    .unwrap();
                unsafe { std::slice::from_raw_parts(h.as_ptr() as *const u16, n).to_vec() }
            };
            // Launch #1 (preflight fires), then #2..#28.
            let first = launch(&fmha);
            let mut snaps = vec![first.clone()];
            for _ in 2..=28 {
                snaps.push(launch(&fmha));
            }
            // LSE readback after the last launch.
            let mut hlse = HostBuffer::alloc(heads as usize * sq_r * 4).unwrap();
            copy(
                &mut MemRef::Host(&mut hlse),
                &mut MemRef::Device(&lse),
                heads as usize * sq_r * 4,
                None,
            )
            .unwrap();
            let lsef: &[f32] = unsafe {
                std::slice::from_raw_parts(hlse.as_ptr() as *const f32, heads as usize * sq_r)
            };
            let worst = |snap: &[u16]| -> f32 {
                (0..n)
                    .map(|i| (f16_to_f32(snap[i]) - f16_to_f32(reff[i])).abs())
                    .fold(0.0f32, f32::max)
            };
            let w1 = worst(&snaps[0]);
            let w28 = worst(&snaps[27]);
            let nbad28 = (0..n)
                .filter(|&i| (f16_to_f32(snaps[27][i]) - f16_to_f32(reff[i])).abs() > 1e-3)
                .count();
            // Per-64-row-band bad count (all heads pooled).
            let mut band_bad = [0usize; 4];
            for s in 0..seq as usize {
                for h in 0..heads as usize {
                    let base = s * nqk + h * d as usize;
                    let bad = (0..d as usize)
                        .filter(|&i| (f16_to_f32(snaps[27][base + i]) - f16_to_f32(reff[base + i])).abs() > 1e-3)
                        .count();
                    if bad > 0 {
                        band_bad[(s / 64).min(3)] += 1;
                    }
                }
            }
            // First bad (s, h) — anatomy: kernel O/LSE vs host.
            let mut first_bad = None;
            'outer: for s in 0..seq as usize {
                for h in 0..heads as usize {
                    let base = s * nqk + h * d as usize;
                    for i in 0..d as usize {
                        if (f16_to_f32(snaps[27][base + i]) - f16_to_f32(reff[base + i])).abs() > 1e-3 {
                            first_bad = Some((s, h));
                            break 'outer;
                        }
                    }
                }
            }
            let mut det = "all-28-identical".to_string();
            for li in 1..28 {
                if snaps[li] != snaps[0] {
                    det = format!("diverges at launch {}", li + 1);
                    break;
                }
            }
            let bad_str = band_bad
                .iter()
                .enumerate()
                .map(|(bi, &c)| format!("b{bi}({}..{}):{c}", bi * 64, bi * 64 + 63))
                .collect::<Vec<_>>()
                .join(" ");
            let anat = match first_bad {
                Some((s, h)) => {
                    let base = s * nqk + h * d as usize;
                    let klse = lsef[h * sq_r + s];
                    format!(
                        "first-bad (s={s},h={h}): O_kernel={:.4e} O_ref={:.4e} LSE_kernel={:.3e} LSE_ref={:.3e}",
                        f16_to_f32(snaps[27][base]),
                        f16_to_f32(reff[base]),
                        klse,
                        reflse[h * sq_r + s]
                    )
                }
                None => "no-bad-rows".to_string(),
            };
            let (aq, ak, av, ao, al) = cell_addr();
            println!(
                "seq={seq} h={heads} kv={kv_heads} {draw}: L1 {w1:.2e} L28 {w28:.2e}                  nbad={nbad28}/{} | {bad_str} | {det} | {anat} | bufs q={aq:x} k={ak:x} v={av:x} o={ao:x} lse={al:x}",
                n
            );
            if draw == "gate_draw" && heads == 8 && kv_heads == 4 && seq == 256 {
                // SAME buffers, DIFFERENT contents — the engine's per-layer
                // case (q/k/v are rewritten every layer, addresses
                // unchanged). If the priming is sticky per buffer set,
                // these launches stay correct WITHOUT a new preflight.
                let (dhq, dhk, dhv) = dense_draw(seq, heads, kv_heads);
                copy(&mut MemRef::Device(&q), &mut MemRef::Host(&dhq), n * 2, None).unwrap();
                copy(&mut MemRef::Device(&k), &mut MemRef::Host(&dhk), nk * 2, None).unwrap();
                copy(&mut MemRef::Device(&v), &mut MemRef::Host(&dhv), nk * 2, None).unwrap();
                let dq32: Vec<f32> = unsafe {
                    std::slice::from_raw_parts(dhq.as_ptr() as *const u16, n)
                        .iter().map(|&b| f16_to_f32(b)).collect()
                };
                let dk32: Vec<f32> = unsafe {
                    std::slice::from_raw_parts(dhk.as_ptr() as *const u16, nk)
                        .iter().map(|&b| f16_to_f32(b)).collect()
                };
                let dv32: Vec<f32> = unsafe {
                    std::slice::from_raw_parts(dhv.as_ptr() as *const u16, nk)
                        .iter().map(|&b| f16_to_f32(b)).collect()
                };
                let (dref, _) = fmha_host_ref(
                    &dq32,
                    &dk32,
                    &dv32,
                    seq as usize,
                    heads as usize,
                    kv_heads as usize,
                    d as usize,
                );
                let snaps2 = (0..28).map(|_| launch(&fmha)).collect::<Vec<_>>();
                let mut w2 = 0.0f32;
                let mut nb2 = 0usize;
                for s in 0..seq as usize {
                    for h in 0..heads as usize {
                        let base = s * nqk + h * d as usize;
                        for i in 0..d as usize {
                            let dv = (f16_to_f32(snaps2[27][base + i]) - f16_to_f32(dref[base + i])).abs();
                            if dv > w2 {
                                w2 = dv;
                            }
                            if dv > 1e-3 {
                                nb2 += 1;
                            }
                        }
                    }
                }
                let det2 = if snaps2.iter().all(|s| s == &snaps2[0]) {
                    "all-28-identical"
                } else {
                    "diverges"
                };
                // Re-draw anatomy: first bad (s, h) + 64-row bands.
                let mut fb2 = None;
                'outer2: for s in 0..seq as usize {
                    for h in 0..heads as usize {
                        let base = s * nqk + h * d as usize;
                        for i in 0..d as usize {
                            if (f16_to_f32(snaps2[27][base + i]) - f16_to_f32(dref[base + i])).abs()
                                > 1e-3
                            {
                                fb2 = Some((s, h));
                                break 'outer2;
                            }
                        }
                    }
                }
                let mut band2 = [0usize; 4];
                for s in 0..seq as usize {
                    for h in 0..heads as usize {
                        let base = s * nqk + h * d as usize;
                        let bad = (0..d as usize)
                            .filter(|&i| {
                                (f16_to_f32(snaps2[27][base + i]) - f16_to_f32(dref[base + i]))
                                    .abs()
                                    > 1e-3
                            })
                            .count();
                        if bad > 0 {
                            band2[(s / 64).min(3)] += 1;
                        }
                    }
                }
                let anat2 = match fb2 {
                    Some((s, h)) => {
                        let base = s * nqk + h * d as usize;
                        format!(
                            "first-bad (s={s},h={h}): kernel={:?} ref={:?}",
                            &snaps2[27][base..base + 4],
                            &dref[base..base + 4]
                        )
                    }
                    None => "no-bad-rows".to_string(),
                };
                println!(
                    "SAME-BUFFER data-change (gate->dense) at seq={seq} h={heads} kv={kv_heads}: L28 {w2:.2e} nbad={nb2}/{} | {det2} | {anat2} | b0(0..63):{} b1(64..127):{} b2(128..191):{} b3(192..255):{}",
                    n,
                    band2[0],
                    band2[1],
                    band2[2],
                    band2[3]
                );
                // Decisive control: run the SAME re-drawn (dense) data
                // through a FRESH FmhaKernels instance on fresh buffers.
                // Its first launch fires the preflight with the dense
                // data (known-good same-data predecessor). If the fresh
                // output reproduces L28, the kernel is deterministic for
                // this data and the inline ref is wrong (probe bug). If
                // it differs (fresh correct, L28 wrong), the kernel is
                // predecessor-dependent -> real latent smem bug.
                let fmha2 = FmhaKernels::new(&arch, Some(cache.clone()), stream.clone()).unwrap();
                let q2 = DeviceBuffer::alloc(devid, n * 2).unwrap();
                let k2 = DeviceBuffer::alloc(devid, nk * 2).unwrap();
                let v2 = DeviceBuffer::alloc(devid, nk * 2).unwrap();
                let o2 = DeviceBuffer::alloc(devid, n * 2).unwrap();
                let lse2 = DeviceBuffer::alloc(devid, heads as usize * sq_r * 4).unwrap();
                copy(&mut MemRef::Device(&q2), &mut MemRef::Host(&dhq), n * 2, None).unwrap();
                copy(&mut MemRef::Device(&k2), &mut MemRef::Host(&dhk), nk * 2, None).unwrap();
                copy(&mut MemRef::Device(&v2), &mut MemRef::Host(&dhv), nk * 2, None).unwrap();
                let launch2 = |fmha: &FmhaKernels| -> Vec<u16> {
                    fmha.launch_batched_prefill(
                        dev,
                        q2.as_ptr() as *const u16,
                        k2.as_ptr() as *const u16,
                        v2.as_ptr() as *const u16,
                        o2.as_ptr() as *mut u16,
                        lse2.as_ptr() as *mut f32,
                        seq,
                        1,
                        heads,
                        kv_heads,
                        d,
                    )
                    .unwrap();
                    stream.synchronize().unwrap();
                    let mut h = HostBuffer::alloc(n * 2).unwrap();
                    copy(&mut MemRef::Host(&mut h), &mut MemRef::Device(&o2), n * 2, None)
                        .unwrap();
                    unsafe { std::slice::from_raw_parts(h.as_ptr() as *const u16, n).to_vec() }
                };
                let fresh = launch2(&fmha2);
                let mut wfresh = 0.0f32;
                let mut nbfresh = 0usize;
                for s in 0..seq as usize {
                    for h in 0..heads as usize {
                        let base = s * nqk + h * d as usize;
                        for i in 0..d as usize {
                            let dv =
                                (f16_to_f32(fresh[base + i]) - f16_to_f32(dref[base + i])).abs();
                            if dv > wfresh {
                                wfresh = dv;
                            }
                            if dv > 1e-3 {
                                nbfresh += 1;
                            }
                        }
                    }
                }
                println!(
                    "FRESH-INSTANCE control (dense data, fresh bufs+preflight): worst {wfresh:.2e} nbad={nbfresh}/{} | same-as-L28: {}",
                    n,
                    fresh == snaps2[27]
                );
                // --- Smem-painter experiments (no memcpy between the
                // painter and the relaunch) ---------------------------
                // Painter A: q=k=v=0x7BFF (max f16) on FRESH buffers.
                // Its v2 launch writes its own tiles into [0,64KiB) of
                // smem (Q region [0,32KiB) ends up holding its O after
                // the epilogue; K/V at [32,64KiB)); the [64,96KiB) tail
                // is only ever written by v0 (the preflight). Then the
                // dense relaunch reads the aliased region:
                //   - Q-fragment aliased  -> scores ~ q*0x7BFF huge,
                //     P ~ uniform -> O ~ mean(dense V)
                //   - K-fragment aliased  -> same uniform signature
                //   - V-fragment aliased  -> O = 0x7BFF EXACTLY everywhere
                let cq = DeviceBuffer::alloc(devid, n * 2).unwrap();
                let ck = DeviceBuffer::alloc(devid, nk * 2).unwrap();
                let cv = DeviceBuffer::alloc(devid, nk * 2).unwrap();
                let co = DeviceBuffer::alloc(devid, n * 2).unwrap();
                let clse = DeviceBuffer::alloc(devid, heads as usize * sq_r * 4).unwrap();
                let ch16 = HostBuffer::alloc(n * 2).unwrap();
                let chk16 = HostBuffer::alloc(nk * 2).unwrap();
                {
                    let s: &mut [u16] = unsafe {
                        std::slice::from_raw_parts_mut(ch16.as_ptr() as *mut u16, n)
                    };
                    s.fill(0x7BFF);
                    let sk: &mut [u16] = unsafe {
                        std::slice::from_raw_parts_mut(chk16.as_ptr() as *mut u16, nk)
                    };
                    sk.fill(0x7BFF);
                }
                copy(&mut MemRef::Device(&cq), &mut MemRef::Host(&ch16), n * 2, None).unwrap();
                copy(&mut MemRef::Device(&ck), &mut MemRef::Host(&chk16), nk * 2, None).unwrap();
                copy(&mut MemRef::Device(&cv), &mut MemRef::Host(&chk16), nk * 2, None).unwrap();
                // Painter-A launch (v2-const; the v0-const preflight
                // fires on this new address key, painting [0,96KiB)).
                fmha.launch_batched_prefill(
                    dev,
                    cq.as_ptr() as *const u16,
                    ck.as_ptr() as *const u16,
                    cv.as_ptr() as *const u16,
                    co.as_ptr() as *mut u16,
                    clse.as_ptr() as *mut f32,
                    seq,
                    1,
                    heads,
                    kv_heads,
                    d,
                )
                .unwrap();
                stream.synchronize().unwrap();
                let pa = launch(&fmha);
                let mut wpa = 0.0f32;
                let mut nbpa = 0usize;
                for s in 0..seq as usize {
                    for h in 0..heads as usize {
                        let base = s * nqk + h * d as usize;
                        for i in 0..d as usize {
                            let dv = (f16_to_f32(pa[base + i]) - f16_to_f32(dref[base + i])).abs();
                            if dv > wpa {
                                wpa = dv;
                            }
                            if dv > 1e-3 {
                                nbpa += 1;
                            }
                        }
                    }
                }
                let row64 = 64usize * nqk;
                println!(
                    "PAINTER-A(0x7BFF)+dense relaunch: worst {wpa:.2e} nbad={nbpa}/{} | same-as-broken-L28: {} | O64[0..4]={:?} ref={:?}",
                    n,
                    pa == snaps2[27],
                    &pa[row64..row64 + 4],
                    &dref[row64..row64 + 4]
                );
                // Painter B: q=k=v=the CURRENT dense data (the gate-cell
                // buffers themselves). Its v2 launch writes [0,64KiB)
                // with the dense data; [64,96KiB) keeps Painter-A's v0
                // const leftovers. If the aliased read lives in [0,64),
                // the relaunch after B must be CORRECT (self-cure). If
                // it lives in [64,96), the relaunch must equal the
                // Painter-A relaunch (const garbage, uncured).
                let pb = launch(&fmha); // the painter itself (dense data)
                let relaunch_after_self = launch(&fmha);
                let mut wsp = 0.0f32;
                let mut nbsp = 0usize;
                for s in 0..seq as usize {
                    for h in 0..heads as usize {
                        let base = s * nqk + h * d as usize;
                        for i in 0..d as usize {
                            let dv = (f16_to_f32(relaunch_after_self[base + i])
                                - f16_to_f32(dref[base + i]))
                            .abs();
                            if dv > wsp {
                                wsp = dv;
                            }
                            if dv > 1e-3 {
                                nbsp += 1;
                            }
                        }
                    }
                }
                println!(
                    "PAINTER-B(self dense)+dense relaunch: worst {wsp:.2e} nbad={nbsp}/{} | same-as-PA: {} | O64[0..4]={:?}",
                    n,
                    relaunch_after_self == pa,
                    &relaunch_after_self[row64..row64 + 4]
                );
                // DECISIVE: fresh FmhaKernels on the GATE-CELL buffers.
                // The buffers still hold the dense data (no memcpy). Its
                // primed set is empty -> the preflight (v0) fires WITH
                // THE DENSE DATA on these EXACT addresses, then the real
                // v2 launch runs. The painters showed smem content is
                // irrelevant; the fresh-buffer control showed the data
                // itself is fine. So this run differs from the broken
                // SAME-BUFFER relaunch ONLY in the v0-same-data
                // predecessor (preflight) on these buffers:
                //   correct -> the preflight itself is the cure (and the
                //     buffers are innocent) -> keep the preflight
                //   broken  -> the addresses themselves are cursed -> the
                //     preflight is a red herring
                let fmha3 = FmhaKernels::new(&arch, Some(cache.clone()), stream.clone()).unwrap();
                fmha3.launch_batched_prefill(
                    dev,
                    q.as_ptr() as *const u16,
                    k.as_ptr() as *const u16,
                    v.as_ptr() as *const u16,
                    o.as_ptr() as *mut u16,
                    lse.as_ptr() as *mut f32,
                    seq,
                    1,
                    heads,
                    kv_heads,
                    d,
                )
                .unwrap();
                stream.synchronize().unwrap();
                let mut fg = HostBuffer::alloc(n * 2).unwrap();
                copy(&mut MemRef::Host(&mut fg), &mut MemRef::Device(&o), n * 2, None).unwrap();
                let fg: Vec<u16> =
                    unsafe { std::slice::from_raw_parts(fg.as_ptr() as *const u16, n).to_vec() };
                let mut wfg = 0.0f32;
                let mut nbfg = 0usize;
                for s in 0..seq as usize {
                    for h in 0..heads as usize {
                        let base = s * nqk + h * d as usize;
                        for i in 0..d as usize {
                            let dv = (f16_to_f32(fg[base + i]) - f16_to_f32(dref[base + i])).abs();
                            if dv > wfg {
                                wfg = dv;
                            }
                            if dv > 1e-3 {
                                nbfg += 1;
                            }
                        }
                    }
                }
                println!(
                    "FRESH-INSTANCE gate-cell bufs (dense, preflight fires): worst {wfg:.2e} nbad={nbfg}/{} | same-as-broken-L28: {} | O64[0..4]={:?}",
                    n,
                    fg == snaps2[27],
                    &fg[row64..row64 + 4]
                );
                // STALE-CHECK: the broken dense relaunch's rows 128..255
                // vs the LAST GATE OUTPUT's rows 128..255. Under the
                // skipped-write model they must be BIT-IDENTICAL (the v2
                // last CTA never writes O; the readback shows the last
                // writer's values = the gate launches').
                let stale_rows = snaps2[27][128 * nqk..256 * nqk] == snaps[27][128 * nqk..256 * nqk];
                let stale_rows_n = (0..128 * nqk)
                    .filter(|&i| snaps2[27][128 * nqk + i] != snaps[27][128 * nqk + i])
                    .count();
                println!(
                    "STALE-CHECK broken-vs-gate rows 128..255: identical={stale_rows} differing={stale_rows_n}/{}",
                    128 * nqk
                );
                // PATTERN-FILL (write-skip proof): fill the ENTIRE o and
                // lse with 0xAAAA (u16) / 0x41414141 (f32) via plain
                // host->device copies — NOT an FMHA launch, so no preflight
                // can fire. Then ONE dense relaunch via the shared
                // instance (key primed). If the v2's last-CTA O/LSE-write
                // is skipped, rows 128..255 come back EXACTLY 0xAAAA (and
                // the lse rows 128..255 EXACTLY the f32 pattern) while
                // rows 0..127 are freshly computed.
                let paaa = HostBuffer::alloc(n * 2).unwrap();
                {
                    let s: &mut [u16] = unsafe {
                        std::slice::from_raw_parts_mut(paaa.as_ptr() as *mut u16, n)
                    };
                    s.fill(0xAAAA);
                }
                let plse = HostBuffer::alloc(heads as usize * sq_r * 4).unwrap();
                {
                    let s: &mut [u32] = unsafe {
                        std::slice::from_raw_parts_mut(plse.as_ptr() as *mut u32, (heads as usize) * sq_r)
                    };
                    s.fill(0x41414141);
                }
                copy(&mut MemRef::Device(&o), &mut MemRef::Host(&paaa), n * 2, None).unwrap();
                copy(
                    &mut MemRef::Device(&lse),
                    &mut MemRef::Host(&plse),
                    heads as usize * sq_r * 4,
                    None,
                )
                .unwrap();
                let pat = launch(&fmha); // dense data still in q/k/v
                let mut wp0 = 0.0f32;
                let mut nb0 = 0usize;
                let mut same_aaa = 0usize;
                for s in 0..seq as usize {
                    for h in 0..heads as usize {
                        let base = s * nqk + h * d as usize;
                        for i in 0..d as usize {
                            let dv =
                                (f16_to_f32(pat[base + i]) - f16_to_f32(dref[base + i])).abs();
                            if dv > wp0 && s < 128 {
                                wp0 = dv;
                            }
                            if dv > 1e-3 && s < 128 {
                                nb0 += 1;
                            }
                        }
                    }
                }
                same_aaa = (128 * nqk..256 * nqk)
                    .filter(|&i| pat[i] == 0xAAAA)
                    .count();
                let mut hlse2 = HostBuffer::alloc(heads as usize * sq_r * 4).unwrap();
                copy(
                    &mut MemRef::Host(&mut hlse2),
                    &mut MemRef::Device(&lse),
                    heads as usize * sq_r * 4,
                    None,
                )
                .unwrap();
                let lse2f: &[u32] = unsafe {
                    std::slice::from_raw_parts(hlse2.as_ptr() as *const u32, (heads as usize) * sq_r)
                };
                // All heads pooled: rows 128..255 across 8 heads.
                let lse_aaa = (0..heads as usize)
                    .flat_map(|hi| (128..256).map(move |s| lse2f[hi * sq_r + s]))
                    .filter(|&x| x == 0x41414141)
                    .count();
                let lse_fresh = (0..heads as usize)
                    .flat_map(|hi| (0..128).map(move |s| lse2f[hi * sq_r + s]))
                    .filter(|&x| x == 0x41414141)
                    .count();
                println!(
                    "PATTERN-FILL 0xAAAA relaunch: rows0..127 worst {wp0:.2e} nbad={nb0} | rows128..255 ==0xAAAA: {same_aaa}/{} | lse rows128..255 ==0x41414141: {lse_aaa}/{} | lse rows0..127 stale-pattern: {lse_fresh}",
                    128 * nqk,
                    8 * 128,
                );
                // Reverse control: re-upload the ORIGINAL gate data into
                // the ORIGINAL buffers and launch once. Its predecessor
                // is the 28th dense launch, so under the smem-leftover
                // model the uninit slots now hold dense data and this
                // relaunch must ALSO be wrong (two-way sensitivity).
                copy(&mut MemRef::Device(&q), &mut MemRef::Host(&hq), n * 2, None).unwrap();
                copy(&mut MemRef::Device(&k), &mut MemRef::Host(&hk), nk * 2, None).unwrap();
                copy(&mut MemRef::Device(&v), &mut MemRef::Host(&hv), nk * 2, None).unwrap();
                let gateback = launch(&fmha);
                let mut wg = 0.0f32;
                let mut nbg = 0usize;
                for s in 0..seq as usize {
                    for h in 0..heads as usize {
                        let base = s * nqk + h * d as usize;
                        for i in 0..d as usize {
                            let dv = (f16_to_f32(gateback[base + i]) - f16_to_f32(reff[base + i]))
                                .abs();
                            if dv > wg {
                                wg = dv;
                            }
                            if dv > 1e-3 {
                                nbg += 1;
                            }
                        }
                    }
                }
                println!("RE-UP gate data same bufs: worst {wg:.2e} nbad={nbg}/{}", n);
            }


            // The closures borrow q/k/v/o/lse; end the borrows before the
            // keepalive move below (the re-draw block above is the last
            // user of `launch`).
            drop(cell_addr);
            drop(launch);
            if keep_this_cell {
                // Keep this cell's buffers alive past the loop: the gate
                // cell below must get FRESH addresses (a dropped buffer's
                // address would be reused by the allocator, colliding the
                // preflight key and silently skipping its preflight).
                keepalive.extend([q, k, v, o, lse]);
            }
        }
    }
}
