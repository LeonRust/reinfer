//! 006 T6: fused Q8_0 dequant-dot decode kernel — real-machine diff vs the
//! existing "dequant -> fp16 -> GEMM" path (003 D7 gate), determinism, and
//! the engine-view per-step latency vs the 003 dense path.
//!
//! Run (any N-card; --test-threads=1 enforced):
//! ```text
//! REINFER_JIT_CACHE=/tmp/reinfer-jit-dequant-dot \
//! REINFER_CUDA_NVCC=/usr/local/cuda-13.2/bin/nvcc \
//! CUDA_VISIBLE_DEVICES=0 \
//! cargo test -p reinfer-cuda --features cuda --test dequant_dot -- \
//!     --ignored --test-threads=1
//! ```
//!
//! Criteria (specs/003-cuda-l0/plan.md D7 GEMM row; Q8_0 slot):
//! - f32-out gate (same judge as 003 gemm_diff): rtol 1e-4 + atol 1e-6 vs a
//!   fixed-order serial-k host reference — applied to BOTH the fused kernel
//!   and the existing dequant->cast_f16->transpose->gemm_f32acc path, so the
//!   two implementations are judged to the same precision tier. Accumulation
//!   semantics match the 003 dense path (fp32, f16-in/f32-out). The fused
//!   kernel keeps dequantized values in f32 registers and rounds them to fp16
//!   with a single RNE rounding — bit-exact with dequant_q8_0 + cast_f32_to_f16
//!   (014 r2) per element;
//! - fused-vs-dense direct difference (max abs / max fp32 ulp) is RECORDED,
//!   not gated: two different fp32 summation orders (stride-32+butterfly vs
//!   cuBLAS tiles) differ by sum-order rounding noise on cancellation-heavy
//!   outputs (observed ~1e-6 absolute at |want| ~ 1e-3, k = 1536);
//! - f16-out tier: fused vs dense both RNE-rounded to fp16 -> <= 1 ulp;
//! - determinism: two launches bit-identical.

#![cfg(feature = "cuda")]
#![allow(clippy::unwrap_used)] // test asserts fail fast

mod gpu {
    use cudarc::cublas::sys as blas;
    use reinfer_core::DeviceId;
    use reinfer_cuda::decode::DecodeDotKernels;
    use reinfer_cuda::dequant::DequantKernels;
    use reinfer_cuda::diff::DiffKernels;
    use reinfer_cuda::gemm::{Gemm, GpuMat};
    use reinfer_cuda::{CudaContext, CudaStream, DeviceBuffer, HostBuffer, MemRef, copy};
    use reinfer_gguf::codes::{dequantize_q8_0, f16_to_f32};
    use reinfer_kernels::refs::matmul_ref;
    use std::ffi::c_void;
    use std::os::raw::c_int;
    use std::time::Instant;

    fn xorshift(seed: &mut u64) -> u64 {
        let mut x = *seed;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *seed = x;
        x
    }

    /// 随机 finite fp16 位（指数域 0..=12 → |v| ≤ 2^-3；真实激活量级，
    /// 含次正规——D7 近零 atol 条款的覆盖对象）。
    fn rand_f16_bits(seed: &mut u64) -> u16 {
        let mant = (xorshift(seed) as u16) & 0x3ff;
        let exp = ((xorshift(seed) as u16) % 0x0d) & 0xf;
        (exp << 10) | mant
    }

    /// 随机 Q8_0 blob [n x k]：scale 指数域 0..=10（有限 f16，含次正规；
    /// 真实权重量级；与 dequant_diff.rs 同的有限值域纪律）。
    fn random_q8_blob(n: usize, k: usize, seed: u64) -> Vec<u8> {
        let blocks = n * k / 32;
        let mut s = seed;
        let mut out = Vec::with_capacity(blocks * 34);
        for _ in 0..blocks {
            let mant = (xorshift(&mut s) as u16) & 0x3ff;
            let exp = ((xorshift(&mut s) as u16) % 0x0b) & 0xf;
            let d = (exp << 10) | mant;
            out.extend_from_slice(&d.to_le_bytes());
            for _ in 0..32 {
                out.push(xorshift(&mut s) as u8);
            }
        }
        out
    }

    /// host f32 → f16 位（RNE；与内核 f32_to_hbits / engine f32_to_f16_bits
    /// 同语义——014 r2：Q8_0 的 f32→f16 必须 RNE 单次舍入）。
    fn rne_f16(f: f32) -> u16 {
        let bits = f.to_bits();
        let sign = (bits >> 16) & 0x8000u32;
        let exp = ((bits >> 23) & 0xff) as i32;
        let man = bits & 0x7f_ffff;
        if exp == 0xff {
            return (sign | 0x7c00 | ((man >> 13) & 0x3ff)) as u16;
        }
        let half_exp = exp - 127 + 15;
        if half_exp <= 0 {
            if half_exp < -10 {
                return sign as u16;
            }
            return (sign | ((man | 0x80_0000) >> (1 - half_exp + 13))) as u16;
        }
        if half_exp >= 31 {
            return (sign | 0x7c00) as u16;
        }
        (sign | ((half_exp as u32) << 10) | (man >> 13)) as u16
    }

    fn upl(dev: DeviceId, host: &[u8]) -> DeviceBuffer {
        let hb = HostBuffer::alloc(host.len()).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(host.as_ptr(), hb.as_ptr() as *mut u8, host.len());
        }
        let db = DeviceBuffer::alloc(dev, host.len()).unwrap();
        copy(&mut MemRef::Device(&db), &MemRef::Host(&hb), host.len(), None).unwrap();
        db
    }

    fn d2h_f32(_dev: DeviceId, db: &DeviceBuffer, n: usize) -> Vec<f32> {
        let hb = HostBuffer::alloc(n * 4).unwrap();
        copy(&mut MemRef::Host(&hb), &MemRef::Device(db), n * 4, None).unwrap();
        // SAFETY：pinned host；n×f32
        unsafe { std::slice::from_raw_parts(hb.as_ptr() as *const f32, n).to_vec() }
    }

    /// 现有 Q8_0 路径的逐元素组装（dequant 核 → cast f16 → 转置 [k×n] →
    /// gemm_f32acc，m=1）：与引擎 gemm1 相同的累积语义（32F-acc）。
    #[allow(clippy::too_many_arguments)] // 参考路径参数矩阵（显式化）
    fn run_reference_path(
        dev: u32,
        stream: &CudaStream,
        dq: &DequantKernels,
        diff: &DiffKernels,
        blas: &Gemm,
        blob_dev: &DeviceBuffer,
        x_dev: &DeviceBuffer,
        n: usize,
        k: usize,
    ) -> Vec<f32> {
        let nblocks = (n * k / 32) as u32;
        let dq_out = DeviceBuffer::alloc(DeviceId::new(0), n * k * 4).unwrap();
        let w16 = DeviceBuffer::alloc(DeviceId::new(0), n * k * 2).unwrap();
        let w16t = DeviceBuffer::alloc(DeviceId::new(0), n * k * 2).unwrap();
        let c = DeviceBuffer::alloc(DeviceId::new(0), n * 4).unwrap();
        dq.launch_dequant_q8_0(dev, blob_dev.as_ptr(), dq_out.as_ptr() as *mut f32, nblocks)
            .unwrap();
        diff.launch_cast_f32_f16(
            dev,
            stream,
            dq_out.as_ptr() as *const f32,
            w16.as_ptr() as *mut u16,
            (n * k) as u32,
        )
        .unwrap();
        diff.launch_transpose_f16(
            dev,
            stream,
            w16.as_ptr() as *const u16,
            w16t.as_ptr() as *mut u16,
            n as u32,
            k as u32,
        )
        .unwrap();
        let amat = GpuMat {
            ptr: x_dev.as_ptr() as *mut c_void,
            dtype: blas::cudaDataType_t::CUDA_R_16F,
            ld: k as c_int,
        };
        let bmat = GpuMat {
            ptr: w16t.as_ptr() as *mut c_void,
            dtype: blas::cudaDataType_t::CUDA_R_16F,
            ld: n as c_int,
        };
        let mut cmat = GpuMat {
            ptr: c.as_ptr() as *mut c_void,
            dtype: blas::cudaDataType_t::CUDA_R_32F,
            ld: 1,
        };
        blas.gemm_f32acc(stream, 1, n as c_int, k as c_int, &amat, &bmat, &mut cmat, 1.0, 0.0)
            .unwrap();
        stream.synchronize().unwrap();
        d2h_f32(DeviceId::new(0), &c, n)
    }

    /// 固定串行 k 序 host 参考（dequant f32 → RNE f16 → matmul_ref）：
    /// sum 顺序差异的记录基线（非 gate）。
    fn host_fixed_order_ref(x16: &[u16], blob: &[u8], n: usize, k: usize) -> Vec<f32> {
        let x: Vec<f32> = x16.iter().map(|v| f16_to_f32(*v)).collect();
        let mut w: Vec<f32> = vec![0.0f32; k * n]; // [k×n] 行主序（gemm B 布局）
        let mut row = vec![0.0f32; k];
        let row_bytes = k / 32 * 34;
        for r in 0..n {
            // dequantize_q8_0 按整个切片处理：只传 r 行那一段字节。
            dequantize_q8_0(&blob[r * row_bytes..(r + 1) * row_bytes], &mut row).unwrap();
            for (i, v) in row.iter().enumerate() {
                w[i * n + r] = f16_to_f32(rne_f16(*v));
            }
        }
        matmul_ref(&x, &w, 1, n, k)
    }

    /// fp32 位距离（同号 ulp 数；异号/非有限 → u64::MAX）。
    fn f32_ulp(a: f32, b: f32) -> u64 {
        if a == b {
            return 0;
        }
        if !a.is_finite() || !b.is_finite() || a.is_sign_negative() != b.is_sign_negative() {
            return u64::MAX;
        }
        (a.to_bits() as i64 - b.to_bits() as i64).unsigned_abs()
    }

    /// fp16 位距离（同号 ulp 数；异号/NaN → u32::MAX）。
    fn f16_ulp(a: f32, b: f32) -> u32 {
        let (ha, hb) = (rne_f16(a), rne_f16(b));
        if ha == hb {
            return 0;
        }
        if (ha ^ hb) & 0x8000 != 0 {
            return u32::MAX;
        }
        (ha as i32 - hb as i32).unsigned_abs()
    }

    /// D7 GEMM 档判定 + 统计：返回 (max_rel, over_gate)。
    fn assert_d7(got: &[f32], want: &[f32], shape: (usize, usize)) -> (f32, usize) {
        assert_eq!(got.len(), want.len(), "shape {shape:?}: length");
        let mut max_rel = 0.0f32;
        let mut bad = 0usize;
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            let diff = (g - w).abs();
            let tol = 1e-6 + 1e-4 * w.abs();
            if diff > tol && bad < 4 {
                eprintln!(
                    "dequant-dot mismatch[{i}] (n,k)={shape:?}: got {g:e} want {w:e} diff {diff:e} tol {tol:e}"
                );
            }
            if diff > tol {
                bad += 1;
            }
            let rel = if w.abs() > 1e-9 { diff / w.abs() } else { diff };
            max_rel = max_rel.max(rel);
        }
        assert_eq!(
            bad,
            0,
            "dequant-dot: {bad}/{} elements over D7 gate (rel 1e-4 + atol 1e-6) at (n,k)={shape:?}; worst rel {max_rel:e}",
            got.len()
        );
        (max_rel, bad)
    }

    #[test]
    #[ignore = "gpu.yml: l3-kernels / dequant-dot"]
    fn fused_q8_dot_diff_vs_dequant_gemm() {
        let ctx = CudaContext::init(DeviceId::new(0)).unwrap();
        let dev = ctx.device_id().index();
        let cache = std::env::temp_dir().join("reinfer-jit-dequant-dot");
        let _ = std::fs::remove_dir_all(&cache);
        let stream = CudaStream::new(DeviceId::new(0)).unwrap();
        let dq = DequantKernels::new(
            &reinfer_cuda::arch::resolve_arch().unwrap(),
            Some(cache.clone()),
            stream.clone(),
        )
        .unwrap();
        let diff = DiffKernels::new(
            &reinfer_cuda::arch::resolve_arch().unwrap(),
            Some(cache.clone()),
            stream.clone(),
        )
        .unwrap();
        let blas = Gemm::new(dev).unwrap();
        let dot = DecodeDotKernels::new(
            &reinfer_cuda::arch::resolve_arch().unwrap(),
            Some(cache.clone()),
            stream.clone(),
        )
        .unwrap();

        // (n, k)：最小块 / 尾块守卫（n%8≠0）+真实 K / 真实 K / K 边界 / 真实 N×K。
        let shapes: &[(usize, usize)] =
            &[(32, 32), (40, 896), (32, 1536), (32, 4096), (1536, 1536), (896, 1536)];
        let mut seed = 0x00D6_u64;
        for &(n, k) in shapes {
            let blob = random_q8_blob(n, k, seed);
            seed = seed.wrapping_mul(0x9E37_79B9).wrapping_add(1);
            let mut xseed = seed;
            let x16: Vec<u16> = (0..k).map(|_| rand_f16_bits(&mut xseed)).collect();
            let x_raw: Vec<u8> = x16.iter().flat_map(|v| v.to_le_bytes()).collect();
            let x_dev = upl(DeviceId::new(0), &x_raw);
            let blob_dev = upl(DeviceId::new(0), &blob);

            // 参考路径（现有 dequant→GEMM 组装；003 dense 逐元素语义）
            let dense =
                run_reference_path(dev, &stream, &dq, &diff, &blas, &blob_dev, &x_dev, n, k);
            // fused 核
            let c_dev = DeviceBuffer::alloc(DeviceId::new(0), n * 4).unwrap();
            dot.launch_fused_q8_dot(
                dev,
                x_dev.as_ptr() as *const u16,
                blob_dev.as_ptr(),
                c_dev.as_ptr() as *mut f32,
                n as u32,
                k as u32,
            )
            .unwrap();
            dot.sync_stream().unwrap();
            let got = d2h_f32(DeviceId::new(0), &c_dev, n);

            // ① D7 gate 判定（与 003 gemm_diff 同构：对固定串行 k 序 host 参考，
            //    f32-out，rel 1e-4 + atol 1e-6）。两侧（fused 与 dense）各自对
            //    规范参考达标 ⇒ 与 003 dense 属于同一精度档；两条不同 fp32 求和
            //    序之间的直接差（②）是记录档的 sum-order 噪声，不是 gate。
            let hwant = host_fixed_order_ref(&x16, &blob, n, k);
            let (max_rel_fused, _) = assert_d7(&got, &hwant, (n, k));
            let (max_rel_dense, _) = assert_d7(&dense, &hwant, (n, k));
            // ② fused vs dense 直接差（记录档：max rel / max fp32 ulp）
            let mut max_diff = 0.0f32;
            let mut max_ulp = 0u64;
            for (g, w) in got.iter().zip(dense.iter()) {
                max_diff = max_diff.max((g - w).abs());
                max_ulp = max_ulp.max(f32_ulp(*g, *w));
            }
            // ③ f16-out 档：双侧 RNE → ≤1 ulp（fused 逐元素 y 与 dense 链
            //    dequant→cast 位等价的直接校验）
            let mut over_1ulp = 0usize;
            let mut max_f16_ulp = 0u32;
            for (g, w) in got.iter().zip(dense.iter()) {
                let u = f16_ulp(*g, *w);
                max_f16_ulp = max_f16_ulp.max(u);
                if u > 1 {
                    over_1ulp += 1;
                }
            }
            assert_eq!(
                over_1ulp, 0,
                "dequant-dot f16-out tier: {over_1ulp}/{} elements over 1 ulp (max {max_f16_ulp}) at (n,k)={n},{k}",
                n
            );
            println!(
                "dequant-dot ok (n={n}, k={k}): D7 gate max rel fused {max_rel_fused:e} / dense {max_rel_dense:e} \
                 (vs serial ref); fused-vs-dense max abs {max_diff:e} max {max_ulp} fp32 ulp; \
                 f16-out max {max_f16_ulp} ulp"
            );
        }
    }

    #[test]
    #[ignore = "gpu.yml: l3-kernels / dequant-dot"]
    fn fused_q8_dot_deterministic() {
        let ctx = CudaContext::init(DeviceId::new(0)).unwrap();
        let dev = ctx.device_id().index();
        let cache = std::env::temp_dir().join("reinfer-jit-dequant-dot-det");
        let _ = std::fs::remove_dir_all(&cache);
        let stream = CudaStream::new(DeviceId::new(0)).unwrap();
        let dot = DecodeDotKernels::new(
            &reinfer_cuda::arch::resolve_arch().unwrap(),
            Some(cache.clone()),
            stream.clone(),
        )
        .unwrap();

        let (n, k) = (1536usize, 4096usize);
        let blob = random_q8_blob(n, k, 0xD0D5);
        let mut xseed = 0xD0D5u64;
        let x16: Vec<u16> = (0..k).map(|_| rand_f16_bits(&mut xseed)).collect();
        let x_raw: Vec<u8> = x16.iter().flat_map(|v| v.to_le_bytes()).collect();
        let x_dev = upl(DeviceId::new(0), &x_raw);
        let blob_dev = upl(DeviceId::new(0), &blob);

        let run = |c_dev: &DeviceBuffer| {
            dot.launch_fused_q8_dot(
                dev,
                x_dev.as_ptr() as *const u16,
                blob_dev.as_ptr(),
                c_dev.as_ptr() as *mut f32,
                n as u32,
                k as u32,
            )
            .unwrap();
            dot.sync_stream().unwrap();
            d2h_f32(DeviceId::new(0), c_dev, n)
        };
        let c1 = DeviceBuffer::alloc(DeviceId::new(0), n * 4).unwrap();
        let c2 = DeviceBuffer::alloc(DeviceId::new(0), n * 4).unwrap();
        let a = run(&c1);
        let b = run(&c2);
        assert_eq!(a, b, "determinism: two launches differ (n={n}, k={k})");
        println!("dequant-dot deterministic: bit-identical across launches (n={n}, k={k})");
    }

    #[test]
    #[ignore = "gpu.yml: l3-kernels / dequant-dot"]
    fn fused_q8_dot_engine_view_step_latency() {
        // 引擎视角（与 decode.rs 内 cudaEvent 计时互补）：每步 1 次 launch +
        // 流同步的墙钟单步延迟。fused = 1 launch；003 dense gemm_f32acc = 1
        // launch；dequant 链（dequant+cast+transpose+gemm）= 4 launch（信息项
        // ——014 D4 按层摊销后不进 decode 循环，此处仅为对照）。
        let ctx = CudaContext::init(DeviceId::new(0)).unwrap();
        let dev = ctx.device_id().index();
        let cache = std::env::temp_dir().join("reinfer-jit-dequant-dot-lat");
        let _ = std::fs::remove_dir_all(&cache);
        let stream = CudaStream::new(DeviceId::new(0)).unwrap();
        let dq = DequantKernels::new(
            &reinfer_cuda::arch::resolve_arch().unwrap(),
            Some(cache.clone()),
            stream.clone(),
        )
        .unwrap();
        let diff = DiffKernels::new(
            &reinfer_cuda::arch::resolve_arch().unwrap(),
            Some(cache.clone()),
            stream.clone(),
        )
        .unwrap();
        let blas = Gemm::new(dev).unwrap();
        let dot = DecodeDotKernels::new(
            &reinfer_cuda::arch::resolve_arch().unwrap(),
            Some(cache.clone()),
            stream.clone(),
        )
        .unwrap();

        let (n, k) = (1536usize, 4096usize);
        let blob = random_q8_blob(n, k, 0x51A7);
        let mut xseed = 0x51A7u64;
        let x16: Vec<u16> = (0..k).map(|_| rand_f16_bits(&mut xseed)).collect();
        let x_raw: Vec<u8> = x16.iter().flat_map(|v| v.to_le_bytes()).collect();
        let x_dev = upl(DeviceId::new(0), &x_raw);
        let blob_dev = upl(DeviceId::new(0), &blob);
        let c_dev = DeviceBuffer::alloc(DeviceId::new(0), n * 4).unwrap();
        // dense B：[k×n] 行主序 f16（dequant→RNE f16→转置；与参考路径同构）
        let mut w16t: Vec<u16> = vec![0u16; k * n];
        let mut row = vec![0.0f32; k];
        let row_bytes = k / 32 * 34;
        for r in 0..n {
            // dequantize_q8_0 按整个切片处理：只传 r 行那一段字节。
            dequantize_q8_0(&blob[r * row_bytes..(r + 1) * row_bytes], &mut row).unwrap();
            for (i, v) in row.iter().enumerate() {
                w16t[i * n + r] = rne_f16(*v);
            }
        }
        let w_raw: Vec<u8> = w16t.iter().flat_map(|v| v.to_le_bytes()).collect();
        let w_dev = upl(DeviceId::new(0), &w_raw);
        let amat = GpuMat {
            ptr: x_dev.as_ptr() as *mut c_void,
            dtype: blas::cudaDataType_t::CUDA_R_16F,
            ld: k as c_int,
        };
        let bmat = GpuMat {
            ptr: w_dev.as_ptr() as *mut c_void,
            dtype: blas::cudaDataType_t::CUDA_R_16F,
            ld: n as c_int,
        };
        let mut cmat = GpuMat {
            ptr: c_dev.as_ptr() as *mut c_void,
            dtype: blas::cudaDataType_t::CUDA_R_32F,
            ld: 1,
        };

        // 预热 + min-of-N（min 去除调度噪声；单步延迟的引擎视图）
        let measure = |mut step: Box<dyn FnMut()>| -> f32 {
            for _ in 0..10 {
                step();
            }
            stream.synchronize().unwrap();
            let mut best = f32::MAX;
            for _ in 0..50 {
                let t0 = Instant::now();
                step();
                stream.synchronize().unwrap();
                best = best.min(t0.elapsed().as_secs_f32() * 1e3);
            }
            best
        };
        let ms_fused = measure(Box::new(|| {
            dot.launch_fused_q8_dot(
                dev,
                x_dev.as_ptr() as *const u16,
                blob_dev.as_ptr(),
                c_dev.as_ptr() as *mut f32,
                n as u32,
                k as u32,
            )
            .unwrap()
        }));
        let ms_dense = measure(Box::new(|| {
            blas.gemm_f32acc(&stream, 1, n as c_int, k as c_int, &amat, &bmat, &mut cmat, 1.0, 0.0)
                .unwrap()
        }));
        let ms_chain = measure(Box::new(|| {
            let nblocks = (n * k / 32) as u32;
            let dq_out = DeviceBuffer::alloc(DeviceId::new(0), n * k * 4).unwrap();
            let w16 = DeviceBuffer::alloc(DeviceId::new(0), n * k * 2).unwrap();
            let w16t = DeviceBuffer::alloc(DeviceId::new(0), n * k * 2).unwrap();
            let c = DeviceBuffer::alloc(DeviceId::new(0), n * 4).unwrap();
            let mut cg = GpuMat {
                ptr: c.as_ptr() as *mut c_void,
                dtype: blas::cudaDataType_t::CUDA_R_32F,
                ld: 1,
            };
            dq.launch_dequant_q8_0(dev, blob_dev.as_ptr(), dq_out.as_ptr() as *mut f32, nblocks)
                .unwrap();
            diff.launch_cast_f32_f16(
                dev,
                &stream,
                dq_out.as_ptr() as *const f32,
                w16.as_ptr() as *mut u16,
                (n * k) as u32,
            )
            .unwrap();
            diff.launch_transpose_f16(
                dev,
                &stream,
                w16.as_ptr() as *const u16,
                w16t.as_ptr() as *mut u16,
                n as u32,
                k as u32,
            )
            .unwrap();
            blas.gemm_f32acc(&stream, 1, n as c_int, k as c_int, &amat, &bmat, &mut cg, 1.0, 0.0)
                .unwrap()
        }));
        eprintln!(
            "engine-view step (n={n}, k={k}): fused {ms_fused:.2} ms/step ({:.0} tok/s); \
             003 dense {ms_dense:.2} ms/step ({:.0} tok/s) -> {:.2}x; \
             dequant-chain {ms_chain:.2} ms/step ({:.0} tok/s) [informational]",
            1e3 / ms_fused,
            1e3 / ms_dense,
            ms_dense / ms_fused,
            1e3 / ms_chain
        );
        assert!(ms_fused > 0.0 && ms_dense > 0.0 && ms_chain > 0.0, "latency: zero elapsed");
    }
}
