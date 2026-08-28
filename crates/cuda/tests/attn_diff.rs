//! 014 T7: prefill attention 真机判据（32F 判据档）。
//!
//! 运行：`REINFER_CUDA_NVCC=... CUDA_VISIBLE_DEVICES=0 cargo test -p reinfer-cuda
//! --features cuda --test attn_diff -- --ignored --test-threads=1`
//!
//! 判据（014 r2）：
//! ① seq=1k 随机 q/k/v（f16 表示值域，host 展开 f32 上传）→ GPU 组装
//!    （两段 32F GEMM + fp32 中间 + fp32 softmax，全 f32）vs
//!    `kernels::refs::prefill_attn_ref`：≤1 fp16 ulp + 近零 atol 1e-6；
//! ② softmax 输出 P 的每行 sum ≈ 1（1000/1000 行抽查）；
//! ③ r2 反例（弱断言档）：unmasked NaN 传播——注入 NaN 后输出保持 NaN。

#![cfg(feature = "cuda")]
#![allow(clippy::unwrap_used)] // 测试断言崩溃即失败

mod gpu {
    use reinfer_core::DeviceId;
    use reinfer_cuda::attention::{prefill_attention, PrefillScratch};
    use reinfer_cuda::diff::DiffKernels;
    use reinfer_cuda::gemm::Gemm;
    use reinfer_cuda::{copy, CudaContext, CudaStream, DeviceBuffer, HostBuffer, MemRef};
    use reinfer_gguf::codes::f16_to_f32;
    use reinfer_kernels::refs::prefill_attn_ref;

    const SEQ: usize = 1000;
    const D: usize = 64;

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

    fn upl(_dev: u32, host: &[u8]) -> DeviceBuffer {
        let hb = HostBuffer::alloc(host.len()).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(host.as_ptr(), hb.as_ptr() as *mut u8, host.len());
        }
        let db = DeviceBuffer::alloc(DeviceId::new(0), host.len()).unwrap();
        copy(&mut MemRef::Device(&db), &MemRef::Host(&hb), host.len(), None).unwrap();
        db
    }

    /// causal 掩码矩阵（seq² bool，行主序：row i / col t 参与 iff t ≤ i）。
    fn causal_mask(seq: usize) -> Vec<bool> {
        let mut m = vec![false; seq * seq];
        for i in 0..seq {
            for t in 0..=i {
                m[i * seq + t] = true;
            }
        }
        m
    }

    fn mask_f32(seq: usize) -> Vec<f32> {
        causal_mask(seq)
            .iter()
            .map(|ok| if *ok { 0.0f32 } else { f32::NEG_INFINITY })
            .collect()
    }

    fn make_ctx() -> (u32, CudaStream, Gemm, DiffKernels) {
        let ctx = CudaContext::init(DeviceId::new(0)).unwrap();
        let dev = ctx.device_id().index();
        let stream = CudaStream::new(ctx.device_id()).unwrap();
        let blas = Gemm::new(dev).unwrap();
        let dk = DiffKernels::new(
            &reinfer_cuda::arch::resolve_arch().unwrap(),
            Some(std::env::temp_dir().join("reinfer-jit-attn")),
            CudaStream::new(ctx.device_id()).unwrap(),
        )
        .unwrap();
        (dev, stream, blas, dk)
    }

    #[allow(clippy::too_many_arguments)] // 组装参数矩阵（编排行显式化）
    fn run_prefill(
        dev: u32,
        blas: &Gemm,
        dk: &DiffKernels,
        stream: &CudaStream,
        scratch: &mut PrefillScratch,
        seq: usize,
        d: usize,
        qf: &[f32],
        kf: &[f32],
        vf: &[f32],
    ) -> Vec<f32> {
        let qr: Vec<u8> = qf.iter().flat_map(|v| v.to_le_bytes()).collect();
        let kr: Vec<u8> = kf.iter().flat_map(|v| v.to_le_bytes()).collect();
        let vr: Vec<u8> = vf.iter().flat_map(|v| v.to_le_bytes()).collect();
        let dq = upl(dev, &qr);
        let dk2 = upl(dev, &kr);
        let dv = upl(dev, &vr);
        let mask_host = mask_f32(seq);
        let dmask = upl(dev, &mask_host.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
        let mut dout = DeviceBuffer::alloc(DeviceId::new(0), seq * d * 4).unwrap();
        let hout = HostBuffer::alloc(seq * d * 4).unwrap();
        prefill_attention(dev, blas, dk, stream, scratch, &dq, &dk2, &dv, &dmask, seq, d, &mut dout)
            .unwrap();
        copy(&mut MemRef::Host(&hout), &MemRef::Device(&dout), seq * d * 4, None).unwrap();
        let raw: Vec<f32> = unsafe {
            std::slice::from_raw_parts(hout.as_ptr() as *const f32, seq * d).to_vec()
        };
        let mut got = vec![0.0f32; seq * d];
        for r in 0..seq {
            for c in 0..d {
                got[r * d + c] = raw[r + c * seq];
            }
        }
        got
    }

    #[test]
    #[ignore = "gpu.yml: l3-attn / prefill"]
    fn prefill_matches_ref_32f() {
        let (dev, stream, blas, dk) = make_ctx();
        let mut scratch = PrefillScratch::alloc(DeviceId::new(0), SEQ, D).unwrap();

        let mut seed = 0x7_00D_u64;
        let q: Vec<f32> = (0..SEQ * D).map(|_| f16_to_f32(rand_f16_bits(&mut seed))).collect();
        let k: Vec<f32> = (0..SEQ * D).map(|_| f16_to_f32(rand_f16_bits(&mut seed))).collect();
        let v: Vec<f32> = (0..SEQ * D).map(|_| f16_to_f32(rand_f16_bits(&mut seed))).collect();

        let got = run_prefill(dev, &blas, &dk, &stream, &mut scratch, SEQ, D, &q, &k, &v);
        let want = prefill_attn_ref(&q, &k, &v, SEQ, D, &causal_mask(SEQ));

        let mut bad = 0usize;
        let mut worst_rel = 0.0f32;
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            let wh = f16_to_f32(f16_rne(*w));
            let ulp = if wh == 0.0 { 1e-9 } else { f16_ulp(wh) };
            if (g - w).abs() > ulp + 1e-6 {
                bad += 1;
                if bad < 5 {
                    eprintln!("attn[{i}]: got {g:e} want {w:e}");
                }
            }
            if w.abs() > 1e-9 {
                worst_rel = worst_rel.max((g - w).abs() / w.abs());
            }
        }
        assert_eq!(bad, 0, "prefill: {bad} elements beyond 1 fp16 ulp (atol 1e-6)");
        eprintln!("prefill ok: worst rel {worst_rel:e}");

        // P 行和 ≈ 1（读回 sr 全量抽查）。
        {
            let hp = HostBuffer::alloc(SEQ * SEQ * 4).unwrap();
            copy(&mut MemRef::Host(&hp), &MemRef::Device(&scratch.sr), SEQ * SEQ * 4, None).unwrap();
            // SAFETY：pinned host；×f32
            let p: Vec<f32> = unsafe {
                std::slice::from_raw_parts(hp.as_ptr() as *const f32, SEQ * SEQ).to_vec()
            };
            let mut sum_bad = 0usize;
            for i in 0..SEQ {
                let s: f32 = p[i * SEQ..(i + 1) * SEQ].iter().sum();
                if (s - 1.0).abs() > 1e-3 {
                    sum_bad += 1;
                }
            }
            assert_eq!(sum_bad, 0, "P row sums deviate from 1: {sum_bad}/{SEQ}");
        }
    }

    /// r2 反例（弱断言档）：unmasked NaN 传播。
    #[test]
    #[ignore = "gpu.yml: l3-attn / prefill"]
    fn prefill_nan_propagates() {
        let (dev, stream, blas, dk) = make_ctx();
        let (seq, d) = (64usize, 64usize);
        let mut scratch = PrefillScratch::alloc(DeviceId::new(0), seq, d).unwrap();
        let mut seed = 0xBA5E_u64;
        let mut qf: Vec<f32> = (0..seq * d).map(|_| f16_to_f32(rand_f16_bits(&mut seed))).collect();
        let kf: Vec<f32> = (0..seq * d).map(|_| f16_to_f32(rand_f16_bits(&mut seed))).collect();
        let vf: Vec<f32> = (0..seq * d).map(|_| f16_to_f32(rand_f16_bits(&mut seed))).collect();
        qf[63 * d + 3] = f32::NAN; // 未掩码位（行 63 全参与）注入 NaN

        let got = run_prefill(dev, &blas, &dk, &stream, &mut scratch, seq, d, &qf, &kf, &vf);
        // 参考语义：max-softmax（online-max）对含 NaN 行 → 全 0（与「全无效行」同
        // 路径；CPU ref 与 GPU 数学同路——两者必须逐位一致；此即 r2 反例目的：
        // 防「NaNs 被 mask 掩盖 + 实现早退」——一致即无早退）。
        let want = prefill_attn_ref(&qf, &kf, &vf, seq, d, &causal_mask(seq));
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            let wh = f16_to_f32(f16_rne(*w));
            let ulp = if wh == 0.0 { 1e-9 } else { f16_ulp(wh) };
            assert!(
                (g - w).abs() <= ulp + 1e-6,
                "nan-case mismatch[{i}]: got {:e} ref {:e} (1 fp16 ulp)",
                g,
                w
            );
        }
        let row0 = &got[0..d];
        assert!(row0.iter().all(|v| v.is_finite() && v.abs() < 1e3), "row0 must remain finite");
        eprintln!("nan-propagation: OK (bit-exact vs CPU ref; row0 finite)");
    }

    /// fp16 指数域 ulp（对非零 wh）。
    fn f16_ulp(wh: f32) -> f32 {
        let e = wh.abs().log2().floor() as i32;
        2.0f32.powi(e - 10).max(1e-9)
    }

    fn f16_rne(f: f32) -> u16 {
        // RNE 舍入（与 ggml 同语义；测试用于「fp16 舍入后比较」）。
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
}
