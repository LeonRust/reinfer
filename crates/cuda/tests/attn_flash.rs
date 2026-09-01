//! 006-2 T2: flash-style decode attention — real-machine criteria
//! (diff / determinism / timing). Trigger: S1-5 suspension clause — the
//! S1-1 profile measured the decode-step attn segment at 14.17 ms/step
//! (63.4%) with kv bandwidth only ~37 us/step (naive kernel <3% bandwidth
//! efficiency). The naive kernel recomputes the full q.k_t dot once per
//! output element (d x redundant QK^T work); the flash kernel (one CTA per
//! (b, q_head), 256 threads, smem scores, fixed-tree reductions) removes
//! the redundancy.
//!
//! Run:
//!   REINFER_CUDA_NVCC=/usr/local/cuda-13.2/bin/nvcc CUDA_VISIBLE_DEVICES=0 \
//!   cargo test -p reinfer-cuda --features cuda --test attn_flash -- \
//!     --ignored --test-threads=1 --nocapture
//!
//! Criteria:
//! ① flash vs host gather reference — D7: f16-out both sides rounded,
//!    compare f32 values within 1 fp16 ulp + atol 1e-6; both the identity
//!    page-table fast path (contiguous reads) and the paged path (random
//!    physical pages) with 0xFFFF (NaN) poison beyond kv_len;
//! ② flash vs naive (decode_step_gqa) on identical inputs — D7 same
//!    tolerance (f16 variant) / rel 1e-4 + atol 1e-6 (f32 parity variant);
//! ③ determinism: two launches bit-identical;
//! ④ timing (record tier): isolation attribution of the S1-1 attn segment
//!    — naive vs flash at kv_len 656 / 1312, plus the other attn-component
//!    kernels (head-norm / rope / kv-write) at Qwen3-0.6B shapes.

#![cfg(feature = "cuda")]
#![allow(clippy::unwrap_used)] // 测试断言崩溃即失败

mod gpu {
    use reinfer_core::DeviceId;
    use reinfer_cuda::decode::DecodeKernels;
    use reinfer_cuda::engine::{DenseKernels, Engine};
    use reinfer_cuda::{CudaContext, CudaStream, DeviceBuffer, HostBuffer, MemRef, copy};
    use reinfer_gguf::codes::f16_to_f32;
    use reinfer_tokenizer::Tokenizer;
    use std::time::Instant;

    fn xorshift(seed: &mut u64) -> u64 {
        let mut x = *seed;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *seed = x;
        x
    }

    fn rand_f16_bits(seed: &mut u64) -> u16 {
        let mant = (xorshift(seed) as u16) & 0x3ff;
        let exp = ((xorshift(seed) as u16) % 0x1e) & 0xf;
        (exp << 10) | mant
    }

    fn upl(dev: u32, host: &[u8]) -> DeviceBuffer {
        let hb = HostBuffer::alloc(host.len()).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(host.as_ptr(), hb.as_ptr() as *mut u8, host.len());
        }
        let db = DeviceBuffer::alloc(DeviceId::new(dev), host.len()).unwrap();
        copy(&mut MemRef::Device(&db), &MemRef::Host(&hb), host.len(), None).unwrap();
        db
    }

    /// K/V row base index of token t (identity fast path vs paged lookup —
    /// mirrors decode_flash_kernels.cu krow_ptr/vrow_ptr semantics).
    fn row_base(
        page: &[u32],
        t: usize,
        block_len: usize,
        kv_heads: usize,
        d: usize,
        identity: bool,
    ) -> usize {
        if identity {
            ((page[0] as usize * block_len + t) * kv_heads) * d
        } else {
            let lp = t / block_len;
            let off = t % block_len;
            ((page[lp] as usize * block_len + off) * kv_heads) * d
        }
    }

    /// Host gather reference (f16 q/kv in; f32 out; serial ascending sums —
    /// same math as decode_gqa_kernels.cu, which is the D7 comparison base).
    #[allow(clippy::too_many_arguments)]
    fn decode_ref(
        q: &[u16],
        page: &[u32],
        kv: &[u16],
        kv_lens: &[u32],
        b: usize,
        qh: usize,
        d: usize,
        block_len: usize,
        kv_ratio: usize,
        total_pages: usize,
        identity: bool,
    ) -> Vec<f32> {
        let kv_heads = qh / kv_ratio;
        let per_tok = kv_heads * d;
        let v_base = total_pages * block_len * per_tok;
        let mut out = vec![0.0f32; b * qh * d];
        for bi in 0..b {
            let kv_len = kv_lens[bi] as usize;
            for h in 0..qh {
                let kv_h = h / kv_ratio;
                let mut s = vec![0.0f32; kv_len];
                for t in 0..kv_len {
                    let base = row_base(page, t, block_len, kv_heads, d, identity) + kv_h * d;
                    let mut acc = 0.0f32;
                    for i in 0..d {
                        acc += f16_to_f32(q[(bi * qh + h) * d + i]) * f16_to_f32(kv[base + i]);
                    }
                    s[t] = acc;
                }
                let maxv = s.iter().copied().reduce(f32::max).unwrap_or(-1e30);
                let sum: f32 = s.iter().map(|x| (x - maxv).exp()).sum();
                let inv = if sum != 0.0 { 1.0 / sum } else { 0.0 };
                for i in 0..d {
                    let mut acc = 0.0f32;
                    for t in 0..kv_len {
                        let p = (s[t] - maxv).exp() * inv;
                        let base = row_base(page, t, block_len, kv_heads, d, identity) + kv_h * d;
                        acc += p * f16_to_f32(kv[v_base + base + i]);
                    }
                    out[(bi * qh + h) * d + i] = acc;
                }
            }
        }
        out
    }

    /// Build a poisoned KV buffer [total_pages][block_len][kv_heads][d] x2
    /// (0xFFFF = NaN beyond the written tokens) with the written tokens
    /// following `page` (identity or random physical).
    fn make_kv(
        seed: &mut u64,
        total_pages: usize,
        block_len: usize,
        kv_heads: usize,
        d: usize,
        kv_len: usize,
        page: &[u32],
        identity: bool,
    ) -> Vec<u16> {
        let per_tok = kv_heads * d;
        let mut kv = vec![0xFFFFu16; total_pages * block_len * per_tok * 2];
        for t in 0..kv_len {
            let base = row_base(page, t, block_len, kv_heads, d, identity);
            for kk in 0..kv_heads {
                for i in 0..d {
                    let idx = base + kk * d + i;
                    kv[idx] = rand_f16_bits(seed);
                    kv[total_pages * block_len * per_tok + idx] = rand_f16_bits(seed);
                }
            }
        }
        kv
    }

    fn f16_bits_of(f: f32) -> u16 {
        let bits = f.to_bits();
        let sign: u16 = ((bits >> 16) & 0x8000) as u16;
        let exp = ((bits >> 23) & 0xff) as i32;
        let man = bits & 0x7f_ffff;
        if exp == 0xff {
            return sign | 0x7c00 | (((man >> 13) & 0x3ff) as u16);
        }
        let half_exp = exp - 127 + 15;
        if half_exp <= 0 {
            if half_exp < -10 {
                return sign;
            }
            let subm = ((man | 0x80_0000) >> (1 - half_exp + 13)) as u16;
            return sign | subm;
        }
        if half_exp >= 31 {
            return sign | 0x7c00;
        }
        sign | ((half_exp as u16) << 10) | ((man >> 13) as u16)
    }

    fn ulp_of(wh: f32) -> f32 {
        let e = wh.abs().log2().floor() as i32;
        2.0f32.powi(e - 10).max(1e-9)
    }

    /// D7 judge: f16-out both sides rounded to f16, compare f32 values with
    /// 1 fp16 ulp + atol 1e-6 (the gqa_diff convention).
    fn check_f16_d7(name: &str, got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len(), "{name}: length mismatch");
        let mut bad = 0usize;
        let mut max_rel = 0.0f32;
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            let wh = f16_to_f32(f16_bits_of(*w));
            let ulp = if wh == 0.0 { 1e-9 } else { ulp_of(wh) };
            let diff = (g - w).abs();
            max_rel = max_rel.max(diff / wh.abs().max(1e-30));
            if diff > ulp + 1e-6 {
                bad += 1;
                if bad <= 4 {
                    eprintln!("{name}: elem {i}: got {g} want {w} (wh {wh}, ulp {ulp})");
                }
            }
        }
        assert_eq!(
            bad,
            0,
            "{name}: {bad}/{n} elements over D7 (1 fp16 ulp) — max rel {max_rel:.3e}",
            n = got.len()
        );
    }

    /// D7 judge for f32 outputs (parity tier; f32 acc — rel 1e-4 + atol 1e-6).
    fn check_f32_d7(name: &str, got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len(), "{name}: length mismatch");
        let mut bad = 0usize;
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            let tol = 1e-6 + 1e-4 * w.abs();
            if (g - w).abs() > tol {
                bad += 1;
                if bad <= 4 {
                    eprintln!("{name}: elem {i}: got {g} want {w}");
                }
            }
        }
        assert_eq!(bad, 0, "{name}: {bad} elements over D7 (rel 1e-4/atol 1e-6)");
    }

    fn arch() -> String {
        reinfer_cuda::arch::resolve_arch().unwrap()
    }

    /// One naive (decode_step_gqa) run — f16 q/out; returns f32 math values.
    #[allow(clippy::too_many_arguments)]
    fn run_naive(
        dk: &DecodeKernels,
        dev: u32,
        q: &[u16],
        page: &[u32],
        kv: &[u16],
        kv_lens: &[u32],
        b: u32,
        qh: u32,
        d: u32,
        block_len: u32,
        kv_ratio: u32,
        max_kv: u32,
        total_pages: u32,
    ) -> Vec<f32> {
        let kv_heads = qh / kv_ratio;
        let dq = upl(dev, &q.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
        let dpage = upl(dev, &page.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
        let dkv = upl(dev, &kv.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
        let dlens = upl(dev, &kv_lens.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
        let dsc = DeviceBuffer::alloc(DeviceId::new(dev), (b * qh * max_kv * 4) as usize).unwrap();
        let dout = DeviceBuffer::alloc(DeviceId::new(dev), (b * qh * d * 2) as usize).unwrap();
        dk.launch_decode_step_gqa(
            dev,
            dq.as_ptr() as *const u16,
            dpage.as_ptr() as *const u32,
            dkv.as_ptr() as *const u16,
            dlens.as_ptr() as *const u32,
            dsc.as_ptr() as *mut f32,
            dout.as_ptr() as *mut u16,
            b,
            qh,
            d,
            block_len,
            kv_ratio,
            kv_heads,
            max_kv,
            total_pages,
        )
        .unwrap();
        dk.sync_stream().unwrap();
        let hb = HostBuffer::alloc((b * qh * d * 2) as usize).unwrap();
        copy(&mut MemRef::Host(&hb), &MemRef::Device(&dout), (b * qh * d * 2) as usize, None)
            .unwrap();
        let raw: Vec<u16> = unsafe {
            std::slice::from_raw_parts(hb.as_ptr() as *const u16, (b * qh * d) as usize).to_vec()
        };
        raw.into_iter().map(f16_to_f32).collect()
    }

    /// One flash run — f16 q/out; returns f32 math values.
    #[allow(clippy::too_many_arguments)]
    fn run_flash(
        dk: &DecodeKernels,
        dev: u32,
        q: &[u16],
        page: &[u32],
        kv: &[u16],
        kv_lens: &[u32],
        b: u32,
        qh: u32,
        d: u32,
        block_len: u32,
        kv_ratio: u32,
        max_kv: u32,
        total_pages: u32,
        identity: u32,
    ) -> Vec<f32> {
        let kv_heads = qh / kv_ratio;
        let dq = upl(dev, &q.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
        let dpage = upl(dev, &page.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
        let dkv = upl(dev, &kv.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
        let dlens = upl(dev, &kv_lens.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
        let dout = DeviceBuffer::alloc(DeviceId::new(dev), (b * qh * d * 2) as usize).unwrap();
        dk.launch_decode_step_gqa_flash(
            dev,
            dq.as_ptr() as *const u16,
            dpage.as_ptr() as *const u32,
            dkv.as_ptr() as *const u16,
            dlens.as_ptr() as *const u32,
            dout.as_ptr() as *mut u16,
            b,
            qh,
            d,
            block_len,
            kv_ratio,
            kv_heads,
            max_kv,
            total_pages,
            identity,
        )
        .unwrap();
        dk.sync_stream().unwrap();
        let hb = HostBuffer::alloc((b * qh * d * 2) as usize).unwrap();
        copy(&mut MemRef::Host(&hb), &MemRef::Device(&dout), (b * qh * d * 2) as usize, None)
            .unwrap();
        let raw: Vec<u16> = unsafe {
            std::slice::from_raw_parts(hb.as_ptr() as *const u16, (b * qh * d) as usize).to_vec()
        };
        raw.into_iter().map(f16_to_f32).collect()
    }

    /// One flash f32 run (parity tier: f32 q/out) — returns f32 values.
    #[allow(clippy::too_many_arguments)]
    fn run_flash_f32(
        dk: &DecodeKernels,
        dev: u32,
        q: &[f32],
        page: &[u32],
        kv: &[u16],
        kv_lens: &[u32],
        b: u32,
        qh: u32,
        d: u32,
        block_len: u32,
        kv_ratio: u32,
        max_kv: u32,
        total_pages: u32,
        identity: u32,
    ) -> Vec<f32> {
        let kv_heads = qh / kv_ratio;
        let dq = upl(dev, &q.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
        let dpage = upl(dev, &page.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
        let dkv = upl(dev, &kv.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
        let dlens = upl(dev, &kv_lens.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
        let dout = DeviceBuffer::alloc(DeviceId::new(dev), (b * qh * d * 4) as usize).unwrap();
        dk.launch_decode_step_gqa_flash_f32(
            dev,
            dq.as_ptr() as *const f32,
            dpage.as_ptr() as *const u32,
            dkv.as_ptr() as *const u16,
            dlens.as_ptr() as *const u32,
            dout.as_ptr() as *mut f32,
            b,
            qh,
            d,
            block_len,
            kv_ratio,
            kv_heads,
            max_kv,
            total_pages,
            identity,
        )
        .unwrap();
        dk.sync_stream().unwrap();
        let hb = HostBuffer::alloc((b * qh * d * 4) as usize).unwrap();
        copy(&mut MemRef::Host(&hb), &MemRef::Device(&dout), (b * qh * d * 4) as usize, None)
            .unwrap();
        unsafe {
            std::slice::from_raw_parts(hb.as_ptr() as *const f32, (b * qh * d) as usize).to_vec()
        }
    }

    /// One naive f32 run (decode_step_gqa_f32) for the parity-tier diff.
    #[allow(clippy::too_many_arguments)]
    fn run_naive_f32(
        dk: &DecodeKernels,
        dev: u32,
        q: &[f32],
        page: &[u32],
        kv: &[u16],
        kv_lens: &[u32],
        b: u32,
        qh: u32,
        d: u32,
        block_len: u32,
        kv_ratio: u32,
        max_kv: u32,
        total_pages: u32,
    ) -> Vec<f32> {
        let kv_heads = qh / kv_ratio;
        let dq = upl(dev, &q.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
        let dpage = upl(dev, &page.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
        let dkv = upl(dev, &kv.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
        let dlens = upl(dev, &kv_lens.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
        let dsc = DeviceBuffer::alloc(DeviceId::new(dev), (b * qh * max_kv * 4) as usize).unwrap();
        let dout = DeviceBuffer::alloc(DeviceId::new(dev), (b * qh * d * 4) as usize).unwrap();
        dk.launch_decode_step_gqa_f32(
            dev,
            dq.as_ptr() as *const f32,
            dpage.as_ptr() as *const u32,
            dkv.as_ptr() as *const u16,
            dlens.as_ptr() as *const u32,
            dsc.as_ptr() as *mut f32,
            dout.as_ptr() as *mut f32,
            b,
            qh,
            d,
            block_len,
            kv_ratio,
            kv_heads,
            max_kv,
            total_pages,
        )
        .unwrap();
        dk.sync_stream().unwrap();
        let hb = HostBuffer::alloc((b * qh * d * 4) as usize).unwrap();
        copy(&mut MemRef::Host(&hb), &MemRef::Device(&dout), (b * qh * d * 4) as usize, None)
            .unwrap();
        unsafe {
            std::slice::from_raw_parts(hb.as_ptr() as *const f32, (b * qh * d) as usize).to_vec()
        }
    }

    /// ① flash vs host gather reference — identity fast path, engine shape.
    #[test]
    #[ignore = "gpu.yml: l3-attn / flash"]
    fn flash_vs_host_ref_identity_d128() {
        let ctx = CudaContext::init(DeviceId::new(0)).unwrap();
        let dev = ctx.device_id().index();
        let cache = std::env::temp_dir().join("reinfer-jit-flash-id");
        let _ = std::fs::remove_dir_all(&cache);
        let dk =
            DecodeKernels::new(&arch(), Some(cache), CudaStream::new(DeviceId::new(0)).unwrap())
                .unwrap();
        let (qh, ratio, bl, max_kv, kv_len, d) =
            (16usize, 2usize, 32usize, 896usize, 656usize, 128usize);
        let kv_heads = qh / ratio;
        // Full pool (engine layout): total_pages = n_layers x pp; the layer's
        // identity table is a 28-entry slice at a non-zero base (exercises
        // the page[0] offset path of the contiguous fast read).
        let total_pages = 896usize;
        let mut seed = 0xA11CE_u64;
        let base_page = 137usize;
        let page: Vec<u32> = (base_page..base_page + max_kv / bl).map(|p| p as u32).collect();
        let kv = make_kv(&mut seed, total_pages, bl, kv_heads, d, kv_len, &page, true);
        let q: Vec<u16> = (0..qh * d).map(|_| rand_f16_bits(&mut seed)).collect();
        let lens = [kv_len as u32];
        let got = run_flash(
            &dk,
            dev,
            &q,
            &page,
            &kv,
            &lens,
            1,
            qh as u32,
            d as u32,
            bl as u32,
            ratio as u32,
            max_kv as u32,
            total_pages as u32,
            1,
        );
        let want = decode_ref(&q, &page, &kv, &lens, 1, qh, d, bl, ratio, total_pages, true);
        check_f16_d7("flash identity d128 kv_len=656", &got, &want);
        // Two-pass determinism at the same shape (bit-identical).
        let got2 = run_flash(
            &dk,
            dev,
            &q,
            &page,
            &kv,
            &lens,
            1,
            qh as u32,
            d as u32,
            bl as u32,
            ratio as u32,
            max_kv as u32,
            total_pages as u32,
            1,
        );
        assert_eq!(got, got2, "flash identity: two launches differ");
    }

    /// ① flash vs host gather reference — paged path, random physical
    /// pages (the 014 T8 GQA trio + the engine d=128 shape), poison beyond
    /// kv_len.
    #[test]
    #[ignore = "gpu.yml: l3-attn / flash"]
    fn flash_vs_host_ref_paged() {
        let ctx = CudaContext::init(DeviceId::new(0)).unwrap();
        let dev = ctx.device_id().index();
        let cache = std::env::temp_dir().join("reinfer-jit-flash-paged");
        let _ = std::fs::remove_dir_all(&cache);
        let dk =
            DecodeKernels::new(&arch(), Some(cache), CudaStream::new(DeviceId::new(0)).unwrap())
                .unwrap();
        let cases: &[(usize, usize, usize, usize, usize, usize)] = &[
            // (qh, kv_ratio, block_len, max_kv, kv_len, d)
            (14, 2, 16, 96, 40, 64),    // 14/2; cross 2-3 pages, partial tail
            (12, 2, 32, 256, 200, 64),  // 12/2; long sequence
            (5, 2, 16, 128, 33, 64),    // 5/2 (non-divisible, floor 2); partial head+tail
            (16, 2, 32, 896, 656, 128), // engine shape, paged path
        ];
        let mut seed = 0x6DE0_u64;
        for &(qh, ratio, bl, max_kv, kv_len, d) in cases {
            let total_pages = max_kv.div_ceil(bl);
            let log_pages = kv_len.div_ceil(bl);
            // Random physical pages (shuffled order across the pool).
            let mut phys: Vec<u32> = (0..log_pages)
                .map(|_| ((xorshift(&mut seed) as usize) % total_pages) as u32)
                .collect();
            // Add a duplicated page id (gather semantics must stay correct).
            if log_pages > 1 {
                phys[log_pages - 1] = phys[0];
            }
            let kv_heads = qh / ratio;
            let kv = make_kv(&mut seed, total_pages, bl, kv_heads, d, kv_len, &phys, false);
            let q: Vec<u16> = (0..qh * d).map(|_| rand_f16_bits(&mut seed)).collect();
            let lens = [kv_len as u32];
            let got = run_flash(
                &dk,
                dev,
                &q,
                &phys,
                &kv,
                &lens,
                1,
                qh as u32,
                d as u32,
                bl as u32,
                ratio as u32,
                max_kv as u32,
                total_pages as u32,
                0,
            );
            let want = decode_ref(&q, &phys, &kv, &lens, 1, qh, d, bl, ratio, total_pages, false);
            check_f16_d7(&format!("flash paged qh={qh} kv_len={kv_len}"), &got, &want);
        }
    }

    /// ② flash vs naive on identical inputs (f16 channel) — identity and
    /// paged paths.
    #[test]
    #[ignore = "gpu.yml: l3-attn / flash"]
    fn flash_vs_naive_f16() {
        let ctx = CudaContext::init(DeviceId::new(0)).unwrap();
        let dev = ctx.device_id().index();
        let cache = std::env::temp_dir().join("reinfer-jit-flash-naive");
        let _ = std::fs::remove_dir_all(&cache);
        let dk =
            DecodeKernels::new(&arch(), Some(cache), CudaStream::new(DeviceId::new(0)).unwrap())
                .unwrap();
        // Engine shape, identity fast path.
        {
            let (qh, ratio, bl, max_kv, kv_len, d) =
                (16usize, 2usize, 32usize, 896usize, 656usize, 128usize);
            let kv_heads = qh / ratio;
            let total_pages = max_kv.div_ceil(bl);
            let mut seed = 0xF1A5_u64;
            let base_page = 0usize;
            let page: Vec<u32> = (base_page..base_page + total_pages).map(|p| p as u32).collect();
            let kv = make_kv(&mut seed, total_pages, bl, kv_heads, d, kv_len, &page, true);
            let q: Vec<u16> = (0..qh * d).map(|_| rand_f16_bits(&mut seed)).collect();
            let lens = [kv_len as u32];
            let na = run_naive(
                &dk,
                dev,
                &q,
                &page,
                &kv,
                &lens,
                1,
                qh as u32,
                d as u32,
                bl as u32,
                ratio as u32,
                max_kv as u32,
                total_pages as u32,
            );
            let fl = run_flash(
                &dk,
                dev,
                &q,
                &page,
                &kv,
                &lens,
                1,
                qh as u32,
                d as u32,
                bl as u32,
                ratio as u32,
                max_kv as u32,
                total_pages as u32,
                1,
            );
            check_f16_d7("flash vs naive (identity, kv_len=656)", &fl, &na);
        }
        // GQA trio shapes, paged path.
        let cases: &[(usize, usize, usize, usize, usize, usize)] =
            &[(14, 2, 16, 96, 40, 64), (12, 2, 32, 256, 200, 64), (5, 2, 16, 128, 33, 64)];
        let mut seed = 0xDE5E_u64;
        for &(qh, ratio, bl, max_kv, kv_len, d) in cases {
            let total_pages = max_kv.div_ceil(bl);
            let log_pages = kv_len.div_ceil(bl);
            let phys: Vec<u32> = (0..log_pages)
                .map(|_| ((xorshift(&mut seed) as usize) % total_pages) as u32)
                .collect();
            let kv_heads = qh / ratio;
            let kv = make_kv(&mut seed, total_pages, bl, kv_heads, d, kv_len, &phys, false);
            let q: Vec<u16> = (0..qh * d).map(|_| rand_f16_bits(&mut seed)).collect();
            let lens = [kv_len as u32];
            let na = run_naive(
                &dk,
                dev,
                &q,
                &phys,
                &kv,
                &lens,
                1,
                qh as u32,
                d as u32,
                bl as u32,
                ratio as u32,
                max_kv as u32,
                total_pages as u32,
            );
            let fl = run_flash(
                &dk,
                dev,
                &q,
                &phys,
                &kv,
                &lens,
                1,
                qh as u32,
                d as u32,
                bl as u32,
                ratio as u32,
                max_kv as u32,
                total_pages as u32,
                0,
            );
            check_f16_d7(&format!("flash vs naive (paged, qh={qh})"), &fl, &na);
        }
    }

    /// ② flash f32 variant vs naive f32 (parity tier) — f32 q/out, f16 KV.
    #[test]
    #[ignore = "gpu.yml: l3-attn / flash"]
    fn flash_vs_naive_f32() {
        let ctx = CudaContext::init(DeviceId::new(0)).unwrap();
        let dev = ctx.device_id().index();
        let cache = std::env::temp_dir().join("reinfer-jit-flash-naive-f32");
        let _ = std::fs::remove_dir_all(&cache);
        let dk =
            DecodeKernels::new(&arch(), Some(cache), CudaStream::new(DeviceId::new(0)).unwrap())
                .unwrap();
        let (qh, ratio, bl, max_kv, kv_len, d) =
            (16usize, 2usize, 32usize, 896usize, 656usize, 128usize);
        let kv_heads = qh / ratio;
        let total_pages = max_kv.div_ceil(bl);
        let mut seed = 0xF32A5_u64;
        let base_page = 3usize;
        let page: Vec<u32> = (base_page..base_page + total_pages).map(|p| p as u32).collect();
        let kv = make_kv(&mut seed, total_pages, bl, kv_heads, d, kv_len, &page, true);
        // f32 q with realistic magnitude (|q| <= 2^-3, finite).
        let q: Vec<f32> = (0..qh * d).map(|_| f16_to_f32(rand_f16_bits(&mut seed)) * 0.9).collect();
        let lens = [kv_len as u32];
        let na = run_naive_f32(
            &dk,
            dev,
            &q,
            &page,
            &kv,
            &lens,
            1,
            qh as u32,
            d as u32,
            bl as u32,
            ratio as u32,
            max_kv as u32,
            total_pages as u32,
        );
        let fl = run_flash_f32(
            &dk,
            dev,
            &q,
            &page,
            &kv,
            &lens,
            1,
            qh as u32,
            d as u32,
            bl as u32,
            ratio as u32,
            max_kv as u32,
            total_pages as u32,
            1,
        );
        check_f32_d7("flash f32 vs naive f32 (identity)", &fl, &na);
    }

    /// ③ determinism: two launches bit-identical (f16 out), paged path.
    #[test]
    #[ignore = "gpu.yml: l3-attn / flash"]
    fn flash_deterministic() {
        let ctx = CudaContext::init(DeviceId::new(0)).unwrap();
        let dev = ctx.device_id().index();
        let cache = std::env::temp_dir().join("reinfer-jit-flash-det");
        let _ = std::fs::remove_dir_all(&cache);
        let dk =
            DecodeKernels::new(&arch(), Some(cache), CudaStream::new(DeviceId::new(0)).unwrap())
                .unwrap();
        let (qh, ratio, bl, max_kv, kv_len, d) =
            (14usize, 2usize, 16usize, 96usize, 40usize, 64usize);
        let total_pages = max_kv.div_ceil(bl);
        let log_pages = kv_len.div_ceil(bl);
        let mut seed = 0xD3A2_u64;
        let phys: Vec<u32> =
            (0..log_pages).map(|_| ((xorshift(&mut seed) as usize) % total_pages) as u32).collect();
        let kv_heads = qh / ratio;
        let kv = make_kv(&mut seed, total_pages, bl, kv_heads, d, kv_len, &phys, false);
        let q: Vec<u16> = (0..qh * d).map(|_| rand_f16_bits(&mut seed)).collect();
        let lens = [kv_len as u32];
        let a = run_flash(
            &dk,
            dev,
            &q,
            &phys,
            &kv,
            &lens,
            1,
            qh as u32,
            d as u32,
            bl as u32,
            ratio as u32,
            max_kv as u32,
            total_pages as u32,
            0,
        );
        let b = run_flash(
            &dk,
            dev,
            &q,
            &phys,
            &kv,
            &lens,
            1,
            qh as u32,
            d as u32,
            bl as u32,
            ratio as u32,
            max_kv as u32,
            total_pages as u32,
            0,
        );
        assert_eq!(a, b, "flash determinism: two launches differ");
    }

    // -------------------------------------------------------------------
    // ④ S1-5 attribution refinement (record tier): isolation timing of the
    // attn-segment components at Qwen3-0.6B shapes. The S1-1 profile shows
    // attn = 14.171 ms/step = 506 us/layer x 28 (kv 2..41 pages); this
    // splits the per-layer budget into the decode kernel vs the small
    // kernels (head-norm q/k, rope q/k, kv-write).
    // -------------------------------------------------------------------

    /// Mean wall time of N streamed launches (`sync` runs after the batch).
    /// CudaStream's raw handle is pub(crate), so stream events are
    /// unavailable from an integration test; wall time is GPU-bound for the
    /// naive kernel and a host-latency UPPER BOUND for the ~us-range kernels
    /// (the authoritative attn-segment number comes from the engine
    /// REINFER_DECODE_PROFILE run).
    fn elapsed_ms(launches: usize, mut sync: impl FnMut(), mut f: impl FnMut()) -> f32 {
        let t0 = std::time::Instant::now();
        for _ in 0..launches {
            f();
        }
        sync();
        t0.elapsed().as_secs_f32() * 1e3 / launches as f32
    }

    fn bincopy(v: &[u16]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    /// Isolation micro-bench: naive vs flash decode at the engine shapes,
    /// plus the other ATTN-segment components. Prints the per-layer split
    /// and the projected 28-layer attn budget (record tier — no gate).
    #[test]
    #[ignore = "gpu.yml: l3-attn / flash-bench"]
    fn attn_segment_attribution() {
        let ctx = CudaContext::init(DeviceId::new(0)).unwrap();
        let dev = ctx.device_id().index();
        let stream = CudaStream::new(DeviceId::new(0)).unwrap();
        let cache = std::env::temp_dir().join("reinfer-jit-flash-bench");
        let _ = std::fs::remove_dir_all(&cache);
        let dk = DecodeKernels::new(
            &arch(),
            Some(cache.clone()),
            CudaStream::new(DeviceId::new(0)).unwrap(),
        )
        .unwrap();
        let dense = DenseKernels::new(&arch(), Some(cache)).unwrap();

        // Pool sized for the 1312-token case: 64 pages x 32 tokens; the
        // flash guard bound (max_kv) is the token cap, 2048 >= 1312.
        let (qh, ratio, bl, d) = (16usize, 2usize, 32usize, 128usize);
        let kv_heads = qh / ratio;
        let total_pages = 64usize;
        let max_kv = 2048u32;
        let mut seed = 0xB4CE_u64;
        let page: Vec<u32> = (0..total_pages).map(|p| p as u32).collect();
        let q: Vec<u16> = (0..qh * d).map(|_| rand_f16_bits(&mut seed)).collect();
        let dq = upl(dev, &bincopy(&q));
        let dpage = upl(dev, &page.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
        let dsc =
            DeviceBuffer::alloc(DeviceId::new(dev), (qh * max_kv as usize * 4) as usize).unwrap();
        let dout16 = DeviceBuffer::alloc(DeviceId::new(dev), (qh * d * 2) as usize).unwrap();
        // Full K+V pool: kv_write's k_region offset is total_pages*bl*per_tok
        // (the decode kernels also read from this pool — contents are
        // irrelevant for timing).
        let kvpool =
            DeviceBuffer::alloc(DeviceId::new(dev), total_pages * bl * kv_heads * d * 2 * 2)
                .unwrap();

        let mut us_naive_656 = 0.0f32;
        let mut us_flash_656 = 0.0f32;
        for kv_len in [656usize, 1312usize] {
            let _ = make_kv(&mut seed, total_pages, bl, kv_heads, d, kv_len, &page, true);
            let lens = [kv_len as u32];
            let dlens = upl(dev, &lens.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());

            let mut run_naive = || {
                dk.launch_decode_step_gqa(
                    dev,
                    dq.as_ptr() as *const u16,
                    dpage.as_ptr() as *const u32,
                    kvpool.as_ptr() as *const u16,
                    dlens.as_ptr() as *const u32,
                    dsc.as_ptr() as *mut f32,
                    dout16.as_ptr() as *mut u16,
                    1,
                    qh as u32,
                    d as u32,
                    bl as u32,
                    ratio as u32,
                    kv_heads as u32,
                    max_kv,
                    total_pages as u32,
                )
                .unwrap();
            };
            let mut run_flash = || {
                dk.launch_decode_step_gqa_flash(
                    dev,
                    dq.as_ptr() as *const u16,
                    dpage.as_ptr() as *const u32,
                    kvpool.as_ptr() as *const u16,
                    dlens.as_ptr() as *const u32,
                    dout16.as_ptr() as *mut u16,
                    1,
                    qh as u32,
                    d as u32,
                    bl as u32,
                    ratio as u32,
                    kv_heads as u32,
                    max_kv,
                    total_pages as u32,
                    1,
                )
                .unwrap();
            };
            for _ in 0..20 {
                run_naive();
            }
            dk.sync_stream().unwrap();
            let iters = 200usize;
            let sync = || dk.sync_stream().unwrap();
            let us_naive = elapsed_ms(iters, sync, &mut run_naive) * 1e3; // per-launch us
            let us_flash = elapsed_ms(iters, sync, &mut run_flash) * 1e3;
            eprintln!(
                "decode kernel (kv_len={kv_len}, d={d}): naive {us_naive:.1} us vs \
                 flash {us_flash:.1} us -> {:.1}x",
                us_naive / us_flash
            );
            if kv_len == 656 {
                us_naive_656 = us_naive;
                us_flash_656 = us_flash;
            }
        }

        // The other ATTN-segment components at engine shapes (kv_len
        // independent; measured once). rms_heads q: grid 16 x 256; k: 8;
        // rope_heads q/k; kv_write per_tok=1024 -> grid 4.
        let mut run_rms_q = || {
            dense
                .launch_rms_heads(
                    dev,
                    &stream,
                    dq.as_ptr() as *const u16,
                    dq.as_ptr() as *mut u16,
                    dq.as_ptr() as *const u16,
                    qh as u32,
                    d as u32,
                    1e-6,
                )
                .unwrap();
        };
        let mut run_rope_q = || {
            dense
                .launch_rope_heads(
                    dev,
                    &stream,
                    dq.as_ptr() as *mut u16,
                    qh as u32,
                    (d / 2) as u32,
                    656,
                    1e6,
                    1.0,
                )
                .unwrap();
        };
        let mut run_kv_write = || {
            dense
                .launch_kv_write(
                    dev,
                    &stream,
                    dq.as_ptr() as *const u16,
                    dq.as_ptr() as *const u16,
                    kvpool.as_ptr() as *mut u16,
                    0,
                    0,
                    bl as u32,
                    kv_heads as u32,
                    d as u32,
                    total_pages as u32,
                )
                .unwrap();
        };
        for _ in 0..20 {
            run_rms_q();
            run_rope_q();
            run_kv_write();
        }
        stream.synchronize().unwrap();
        let iters = 300usize;
        let sync = || stream.synchronize().unwrap();
        let us_rms = elapsed_ms(iters, sync, &mut run_rms_q) * 1e3; // per-launch us
        let us_rope = elapsed_ms(iters, sync, &mut run_rope_q) * 1e3;
        let us_kvw = elapsed_ms(iters, sync, &mut run_kv_write) * 1e3;
        // Per-layer attn budget (kv_len=656): decode + rms q/k + rope q/k +
        // kv_write. rms_k/rope_k run at kv_heads=8 rows -> ~half of the
        // 16-row q passes (grid rows scale the work).
        let small = 1.5 * (us_rms + us_rope) + us_kvw;
        let naive_layer = us_naive_656 + small;
        let flash_layer = us_flash_656 + small;
        eprintln!(
            "attn small kernels (Qwen3-0.6B shapes): rms_heads(q shape) {us_rms:.1} us, \
             rope_heads(q shape) {us_rope:.1} us, kv_write {us_kvw:.1} us"
        );
        eprintln!(
            "per-layer attn (kv_len=656): naive {naive_layer:.1} us (28L -> {:.2} ms) vs \
             flash {flash_layer:.1} us (28L -> {:.2} ms); S1-1 measured 506 us/L \
             (14.17 ms/28); budget 4 ms",
            naive_layer * 28.0 / 1e3,
            flash_layer * 28.0 / 1e3
        );
        assert!(us_naive_656 > 0.0 && us_flash_656 > 0.0, "bench: zero elapsed");
    }

    /// ⑤ (acceptance) engine decode profile at the 656-kv-token condition:
    /// 616-token prompt + 40 greedy decode tokens -> steps 21-40 run at
    /// kv_len 637..656 (mean ~646), the S1-5 target window. Run with
    /// REINFER_DECODE_PROFILE=1 to print the segment table at steps 20/40
    /// (the before/after surface — pre-change attn 14.138 ms/step at the
    /// same profiler; budget 4 ms). Record tier; tok/s printed.
    #[test]
    #[ignore = "gpu.yml: l3-attn / flash-profile"]
    fn engine_profile_656kv() {
        let ctx = CudaContext::init(DeviceId::new(0)).unwrap();
        let dev = ctx.device_id();
        let model_dir = std::path::PathBuf::from(
            std::env::var("REINFER_MODEL_DIR").expect("REINFER_MODEL_DIR"),
        );
        let tok: serde_json::Value = serde_json::from_slice(
            &std::fs::read(model_dir.join("tokenizer.json")).expect("tokenizer.json"),
        )
        .expect("tokenizer json");
        let tokcfg: serde_json::Value = serde_json::from_slice(
            &std::fs::read(model_dir.join("tokenizer_config.json")).expect("tokenizer_config.json"),
        )
        .expect("tokcfg json");
        let tokenizer = Tokenizer::from_hf_json(&tok, &tokcfg).expect("hf tokenizer");
        let mut engine = Engine::load(
            dev,
            &reinfer_cuda::arch::resolve_arch().unwrap(),
            Some(std::env::temp_dir().join("reinfer-jit-flash-prof")),
            &model_dir,
            4096,
        )
        .expect("engine load");
        let mut prompt: Vec<u32> = Vec::new();
        while prompt.len() < 616 {
            prompt.extend(
                tokenizer
                    .encode("The quick brown fox jumps over the lazy dog. ", false)
                    .expect("encode"),
            );
        }
        prompt.truncate(616);
        let eos = engine.config().eos_id;
        let n = 40u32;
        let t0 = Instant::now();
        let out = engine.generate(&prompt, n, eos, 0.0).expect("generate");
        let dt = t0.elapsed();
        let steps = out.len().max(1);
        println!(
            "decode(656kv): {steps} tokens in {dt:?} (tpot {:.1} ms/tok, ~{:.1} tok/s)",
            dt.as_secs_f64() * 1000.0 / steps as f64,
            steps as f64 / dt.as_secs_f64()
        );
        assert!(!out.is_empty());
    }

    /// ⑥ (acceptance) text consistency: greedy 16-token double pass — the
    /// flash tier engine vs the naive tier engine (REINFER_DECODE_FLASH=off,
    /// same weights/kv/stream protocol) must produce identical token
    /// sequences; the flash side must show zero fallbacks. Also prints the
    /// 128-token whole-machine decode rate (tok/s, record tier).
    #[test]
    #[ignore = "gpu.yml: l3-attn / flash-text"]
    fn flash_vs_naive_text_and_tok_s() {
        let ctx = CudaContext::init(DeviceId::new(0)).unwrap();
        let dev = ctx.device_id();
        let model_dir = std::path::PathBuf::from(
            std::env::var("REINFER_MODEL_DIR").expect("REINFER_MODEL_DIR"),
        );
        let tok: serde_json::Value = serde_json::from_slice(
            &std::fs::read(model_dir.join("tokenizer.json")).expect("tokenizer.json"),
        )
        .expect("tokenizer json");
        let tokcfg: serde_json::Value = serde_json::from_slice(
            &std::fs::read(model_dir.join("tokenizer_config.json")).expect("tokenizer_config.json"),
        )
        .expect("tokcfg json");
        let tokenizer = Tokenizer::from_hf_json(&tok, &tokcfg).expect("hf tokenizer");
        let arch = reinfer_cuda::arch::resolve_arch().unwrap();

        // Flash tier engine (default env).
        let mut flash_eng = Engine::load(
            dev,
            &arch,
            Some(std::env::temp_dir().join("reinfer-jit-flash-text")),
            &model_dir,
            4096,
        )
        .expect("flash engine load");
        // Naive tier engine — same cache dir, REINFER_DECODE_FLASH=off routes
        // the attn segment to decode_step_gqa (explicit disable, no counter).
        // (edition 2024: env mutation is unsafe.)
        unsafe {
            std::env::set_var("REINFER_DECODE_FLASH", "off");
        }
        let mut naive_eng = Engine::load(
            dev,
            &arch,
            Some(std::env::temp_dir().join("reinfer-jit-flash-text")),
            &model_dir,
            4096,
        )
        .expect("naive engine load");
        unsafe {
            std::env::remove_var("REINFER_DECODE_FLASH");
        }

        let prompt: Vec<u32> = tokenizer.encode("Hello", false).expect("encode");
        let eos = flash_eng.config().eos_id;
        let n = 16u32;
        let flash_out = flash_eng.generate(&prompt, n, eos, 0.0).expect("flash pass");
        assert_eq!(flash_eng.decode_flash_fallbacks(), 0, "flash tier must not fall back");
        let naive_out = naive_eng.generate(&prompt, n, eos, 0.0).expect("naive pass");
        assert_eq!(flash_out, naive_out, "flash vs naive greedy 16 tokens must match");
        let text = tokenizer.decode_all(&flash_out);
        println!("text consistency: flash == naive == {text:?} ({:?})", flash_out);

        // Whole-machine 128-token decode rate on the flash tier.
        let t0 = Instant::now();
        let out = flash_eng.generate(&prompt, 128, eos, 0.0).expect("tok/s pass");
        let dt = t0.elapsed();
        let steps = out.len().max(1);
        println!(
            "flash tier 128-token decode: {steps} tokens in {dt:?} (tpot {:.1} ms/tok, ~{:.1} tok/s)",
            dt.as_secs_f64() * 1000.0 / steps as f64,
            steps as f64 / dt.as_secs_f64()
        );
        assert!(!out.is_empty());
    }
}
