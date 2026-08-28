//! 014 T6: cuBLAS GEMM 真机判据（Vendor 首件）。
//!
//! 运行：`CUDA_VISIBLE_DEVICES=0 cargo test -p reinfer-cuda --features cuda
//! --test gemm_diff -- --ignored --test-threads=1`（需 cublas：workspace
//! cudarc 增 `cublas` feature——014 T6 接线点）。
//!
//! 判据矩阵（014 r2）：
//! ① 门禁档 `CUBLAS_COMPUTE_32F`：f16-in/f32-out 与 f32-in/f32-out 均
//!    **rtol 1e-4 + atol 1e-6**（vs `kernels::matmul_ref` fp32 累加参考；
//!    形状含模型真实 K（896/1536）与 K∈1..4096、含矩形 m≠n）；
//! ② 记录档 `CUBLAS_COMPUTE_16F`：rel 统计打印（非 gate——r1 实测教训
//!    ：16F-acc vs fp32 参考在真实 K 下 92-98% 超差，只声明不立闸）。
//!
//! 行主序 → cuBLAS 列主序映射（gemm-f32acc 已固定 OP_T/OP_T）：
//! A 视为 col-major A^T（[k,m]，ld=k）、B 同（[n,k]，ld=k）；
//! 输出 col-major C'（[m,n]，ldc=m）→ `want[r*n+c] = raw[r + c*m]`。

#![cfg(feature = "cuda")]
#![allow(clippy::unwrap_used)] // 测试断言崩溃即失败

mod gpu {
    use cudarc::cublas::sys as blas;
    use reinfer_core::DeviceId;
    use reinfer_cuda::gemm::{Gemm, GpuMat};
    use reinfer_cuda::{copy, CudaContext, CudaStream, DeviceBuffer, HostBuffer, MemRef};
    use reinfer_gguf::codes::f16_to_f32;
    use reinfer_kernels::refs::matmul_ref;
    use std::ffi::c_void;

    fn xorshift(seed: &mut u64) -> u64 {
        let mut x = *seed;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *seed = x;
        x
    }

    /// 随机 finite fp16 位模式（指数域 0..=30）。
    fn rand_f16_bits(seed: &mut u64) -> u16 {
        let mant = (xorshift(seed) as u16) & 0x3ff;
        let exp = ((xorshift(seed) as u16) % 0x1e) & 0xf;
        (exp << 10) | mant
    }

    fn rand_f32(seed: &mut u64) -> f32 {
        let raw = xorshift(seed) as u32 & 0x3f_ff_ff_ff; // 指数域 0..=126（有限）
        f32::from_bits(raw)
    }

    fn upl(dev: u32, host: &[u8]) -> DeviceBuffer {
        let hb = HostBuffer::alloc(host.len()).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(host.as_ptr(), hb.as_ptr() as *mut u8, host.len());
        }
        let db = DeviceBuffer::alloc(DeviceId::new(0), host.len()).unwrap();
        copy(&mut MemRef::Device(&db), &MemRef::Host(&hb), host.len(), None).unwrap();
        db
    }

    fn d2h(dev: u32, bytes: usize) -> (DeviceBuffer, HostBuffer) {
        let db = DeviceBuffer::alloc(DeviceId::new(0), bytes).unwrap();
        let hb = HostBuffer::alloc(bytes).unwrap();
        (db, hb)
    }

    fn mat(ptr: *mut c_void, dtype: blas::cudaDataType_t, ld: i32) -> GpuMat {
        GpuMat { ptr, dtype, ld }
    }

    fn assert_rtol_atol(got: &[f32], want: &[f32], rtol: f32, atol: f32) {
        assert_eq!(got.len(), want.len());
        let mut worst = 0.0f32;
        let mut bad = 0usize;
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            let diff = (g - w).abs();
            let tol = atol + rtol * w.abs();
            if diff > tol {
                bad += 1;
                if bad < 4 {
                    eprintln!("gemm mismatch[{i}]: got {g:e} want {w:e} diff {diff:e} tol {tol:e}");
                }
            }
            let rel = if w.abs() > 1e-9 { diff / w.abs() } else { diff };
            worst = worst.max(rel);
        }
        assert_eq!(
            bad,
            0,
            "gemm: {bad}/{} elements over rtol {rtol}+atol {atol}; worst rel {worst:e}",
            got.len()
        );
        eprintln!("gemm ok: worst rel {worst:e}");
    }

    /// 跑一次 gemm（f32acc 语义见文件头注）并返回行主序结果。
    #[allow(clippy::too_many_arguments)]
    fn run_gemm_f32acc(
        dev: u32,
        blas: &Gemm,
        stream: &CudaStream,
        m: usize,
        n: usize,
        k: usize,
        a_dev: &DeviceBuffer,
        b_dev: &DeviceBuffer,
        a_dtype: blas::cudaDataType_t,
        out_rows: usize,
        out_cols: usize,
        compute16: bool,
    ) -> Vec<f32> {
        let (dc, hc) = d2h(dev, m * n * if compute16 { 2 } else { 4 });
        let c_dtype = if compute16 {
            blas::cudaDataType_t::CUDA_R_16F
        } else {
            blas::cudaDataType_t::CUDA_R_32F
        };
        let mut cmat = mat(dc.as_ptr() as *mut c_void, c_dtype, m as i32);
        let amat = mat(a_dev.as_ptr() as *mut c_void, a_dtype, k as i32);
        let bmat = mat(b_dev.as_ptr() as *mut c_void, a_dtype, n as i32); // B^T [n x k] col-major -> leading dim = n
        let r = if compute16 {
            blas.gemm_f16_16acc(stream, m as i32, n as i32, k as i32, &amat, &bmat, &mut cmat, 1.0, 0.0)
        } else {
            blas.gemm_f32acc(stream, m as i32, n as i32, k as i32, &amat, &bmat, &mut cmat, 1.0, 0.0)
        };
        r.unwrap();
        stream.synchronize().unwrap();
        let bytes = m * n * if compute16 { 2 } else { 4 };
        copy(&mut MemRef::Host(&hc), &MemRef::Device(&dc), bytes, None).unwrap();
        let raw_f32: Vec<f32> = if compute16 {
            // SAFETY：pinned host；u16×2 → f16→f32（RNE 位构造语义与 codes 一致）
            let u: Vec<u16> =
                unsafe { std::slice::from_raw_parts(hc.as_ptr() as *const u16, m * n).to_vec() };
            u.iter().map(|v| f16_to_f32(*v)).collect()
        } else {
            // SAFETY：pinned host；×f32
            unsafe { std::slice::from_raw_parts(hc.as_ptr() as *const f32, m * n).to_vec() }
        };
        let mut want = vec![0.0f32; out_rows * out_cols];
        for r in 0..out_rows {
            for c in 0..out_cols {
                want[r * out_cols + c] = raw_f32[r + c * m];
            }
        }
        want
    }

    #[test]
    #[ignore = "gpu.yml: l3-f32acc / gemm"]
    fn gemm_f32acc_gates() {
        let ctx = CudaContext::init(DeviceId::new(0)).unwrap();
        let dev = ctx.device_id().index();
        let stream = CudaStream::new(ctx.device_id()).unwrap();
        let blas = Gemm::new(dev).unwrap();

        let shapes: &[(usize, usize, usize)] = &[
            (32, 32, 32),
            (64, 48, 96),
            (128, 128, 256),
            (64, 32, 896),  // 模型真实 K（矩形 m≠n）
            (32, 64, 1536), // 模型真实 K
            (256, 256, 512),
        ];
        let mut seed = 0xA8_1E_u64;
        for &(m, n, k) in shapes {
            // f16-in/f32-out
            let a16: Vec<u16> = (0..m * k).map(|_| rand_f16_bits(&mut seed)).collect();
            let b16: Vec<u16> = (0..k * n).map(|_| rand_f16_bits(&mut seed)).collect();
            let a: Vec<f32> = a16.iter().map(|u| f16_to_f32(*u)).collect();
            let b: Vec<f32> = b16.iter().map(|u| f16_to_f32(*u)).collect();
            let want = matmul_ref(&a, &b, m, n, k);
            let a_raw: Vec<u8> = a16.iter().flat_map(|v| v.to_le_bytes()).collect();
            let b_raw: Vec<u8> = b16.iter().flat_map(|v| v.to_le_bytes()).collect();
            let da = upl(dev, &a_raw);
            let db = upl(dev, &b_raw);
            let got = run_gemm_f32acc(dev, &blas, &stream, m, n, k, &da, &db,
                blas::cudaDataType_t::CUDA_R_16F, m, n, false);
            assert_rtol_atol(&got, &want, 1e-4, 1e-6);

            // f32-in/f32-out
            let af: Vec<f32> = (0..m * k).map(|_| rand_f32(&mut seed)).collect();
            let bf: Vec<f32> = (0..k * n).map(|_| rand_f32(&mut seed)).collect();
            let wantf = matmul_ref(&af, &bf, m, n, k);
            let af_raw: Vec<u8> = af.iter().flat_map(|v| v.to_le_bytes()).collect();
            let bf_raw: Vec<u8> = bf.iter().flat_map(|v| v.to_le_bytes()).collect();
            let daf = upl(dev, &af_raw);
            let dbf = upl(dev, &bf_raw);
            let gotf = run_gemm_f32acc(dev, &blas, &stream, m, n, k, &daf, &dbf,
                blas::cudaDataType_t::CUDA_R_32F, m, n, false);
            assert_rtol_atol(&gotf, &wantf, 1e-4, 1e-6);
        }

        // K 覆盖（1..4096 抽样——判据范围的边界档）
        for &k in &[1usize, 16, 4096] {
            let (m, n) = (32usize, 32usize);
            let a16: Vec<u16> = (0..m * k).map(|_| rand_f16_bits(&mut seed)).collect();
            let b16: Vec<u16> = (0..k * n).map(|_| rand_f16_bits(&mut seed)).collect();
            let a: Vec<f32> = a16.iter().map(|u| f16_to_f32(*u)).collect();
            let b: Vec<f32> = b16.iter().map(|u| f16_to_f32(*u)).collect();
            let want = matmul_ref(&a, &b, m, n, k);
            let a_raw: Vec<u8> = a16.iter().flat_map(|v| v.to_le_bytes()).collect();
            let b_raw: Vec<u8> = b16.iter().flat_map(|v| v.to_le_bytes()).collect();
            let da = upl(dev, &a_raw);
            let db = upl(dev, &b_raw);
            let got = run_gemm_f32acc(dev, &blas, &stream, m, n, k, &da, &db,
                blas::cudaDataType_t::CUDA_R_16F, m, n, false);
            assert_rtol_atol(&got, &want, 1e-4, 1e-6);
        }
    }

    #[test]
    #[ignore = "gpu.yml: l3-f16acc / gemm"]
    fn gemm_f16_16acc_record() {
        // 记录档：compute=16F 的 rel 统计产出（非 gate——只打印+断言宽松 ≤1e-1 声明）
        let ctx = CudaContext::init(DeviceId::new(0)).unwrap();
        let dev = ctx.device_id().index();
        let stream = CudaStream::new(ctx.device_id()).unwrap();
        let blas = Gemm::new(dev).unwrap();
        let (m, n, k) = (32usize, 32usize, 896usize); // 模型真实 K
        let mut seed = 0x1F_16_u64;
        let a16: Vec<u16> = (0..m * k).map(|_| rand_f16_bits(&mut seed)).collect();
        let b16: Vec<u16> = (0..k * n).map(|_| rand_f16_bits(&mut seed)).collect();
        let a: Vec<f32> = a16.iter().map(|u| f16_to_f32(*u)).collect();
        let b: Vec<f32> = b16.iter().map(|u| f16_to_f32(*u)).collect();
        let want = matmul_ref(&a, &b, m, n, k);
        let a_raw: Vec<u8> = a16.iter().flat_map(|v| v.to_le_bytes()).collect();
        let b_raw: Vec<u8> = b16.iter().flat_map(|v| v.to_le_bytes()).collect();
        let da = upl(dev, &a_raw);
        let db = upl(dev, &b_raw);
        let got = run_gemm_f32acc(dev, &blas, &stream, m, n, k, &da, &db,
            blas::cudaDataType_t::CUDA_R_16F, m, n, true);
        let mut max_rel = 0.0f32;
        let mut over_1e_1 = 0usize;
        for (g, w) in got.iter().zip(want.iter()) {
            let rel = if w.abs() > 1e-9 { (g - w).abs() / w.abs() } else { (g - w).abs() };
            max_rel = max_rel.max(rel);
            if rel > 1e-1 {
                over_1e_1 += 1;
            }
        }
        // 声明（r1）：16F-acc 为记录项；rel ≤1e-1 为文档声明（低于此即常态）
        eprintln!("16F-acc record: max rel {max_rel:e}, over 1e-1: {over_1e_1}/{}",
            got.len());
        assert!(
            max_rel <= 1e-1,
            "16F-acc record: max rel {max_rel:e} — unexpected, record-sample broke declared bound"
        );
    }
}
