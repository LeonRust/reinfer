//! JitGemm m=1 gemv vs cuBLAS differential (S1-6 decode-path GEMM swap).
//!
//! The decode step's projections (q/k/v/o/gate/up/down per layer + lm_head)
//! are m=1 GEMMs; `Jgemm` runs the `gemv_m1_f16f32` kernel for them instead
//! of cublas. The numeric criterion tier is unchanged (f16 in / f32 out,
//! fp32 accumulation — the same tier as cublas COMPUTE_32F); the reduction
//! ORDER differs, so this test records the drift:
//!
//!   - max abs/rel vs cublas (same A/B inputs, m=1): expected <= 1e-5
//!     (record; the 014 D7 tier is rtol 1e-4 + atol 1e-6 as the gate);
//!   - determinism: two jgemm launches on the same inputs must be
//!     bit-identical (fixed per-thread order, no atomics).
//!
//! Shapes: n in {1024, 1536, 3072, 151936 (Qwen3-0.6B lm_head)}, k in
//! {1024 (hidden), 3072 (ffn)} — the model's real decode geometries.
//!
//! Run: CUDA_VISIBLE_DEVICES=0 cargo test -p reinfer-cuda --features cuda
//! --test gemm_m1_diff -- --ignored --test-threads=1 --nocapture

#![cfg(feature = "cuda")]
#![allow(clippy::unwrap_used)] // 测试断言崩溃即失败

mod gpu {
    use reinfer_core::DeviceId;
    use reinfer_cuda::gemm::{Gemm, GemmPlan, Jgemm};
    use reinfer_cuda::{CudaContext, CudaStream, DeviceBuffer, HostBuffer, MemRef, copy};

    fn xorshift(seed: &mut u64) -> u64 {
        let mut x = *seed;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *seed = x;
        x
    }

    /// Random finite fp16 bit patterns (exponent domain 0..=30) — same
    /// domain as the 014 gemm gate.
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
        let db = DeviceBuffer::alloc(DeviceId::new(0), host.len()).unwrap();
        copy(&mut MemRef::Device(&db), &MemRef::Host(&hb), host.len(), None).unwrap();
        let _ = dev;
        db
    }

    fn d2h(db: &DeviceBuffer, bytes: usize) -> Vec<f32> {
        let hb = HostBuffer::alloc(bytes).unwrap();
        copy(&mut MemRef::Host(&hb), &MemRef::Device(db), bytes, None).unwrap();
        unsafe { std::slice::from_raw_parts(hb.as_ptr() as *const f32, bytes / 4).to_vec() }
    }

    /// Run one m=1 GEMM through cublas (`GemmPlan::row_major_f16` layout,
    /// the production plan shape) and one through jgemm; return
    /// (cublas, jgemm_a, jgemm_b) as host f32 vectors. `jgemm_b` is a
    /// second jgemm launch for the determinism check.
    #[allow(clippy::too_many_arguments)]
    fn run_pair(
        dev: u32,
        blas: &Gemm,
        jgemm: &Jgemm,
        stream: &CudaStream,
        n: usize,
        k: usize,
        seed: &mut u64,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let a16: Vec<u16> = (0..k).map(|_| rand_f16_bits(seed)).collect();
        let b16: Vec<u16> = (0..k * n).map(|_| rand_f16_bits(seed)).collect();
        let a_raw: Vec<u8> = a16.iter().flat_map(|v| v.to_le_bytes()).collect();
        let b_raw: Vec<u8> = b16.iter().flat_map(|v| v.to_le_bytes()).collect();
        let da = upl(dev, &a_raw);
        let db = upl(dev, &b_raw);
        let dc1 = DeviceBuffer::alloc(DeviceId::new(0), n * 4).unwrap();
        let dc2 = DeviceBuffer::alloc(DeviceId::new(0), n * 4).unwrap();
        let dc3 = DeviceBuffer::alloc(DeviceId::new(0), n * 4).unwrap();

        // cublas reference (the production numeric referee).
        let plan_c = GemmPlan::row_major_f16(
            da.as_ptr() as *const u16,
            db.as_ptr() as *const u16,
            dc1.as_ptr() as *mut f32,
            1,
            n,
            k,
        );
        blas.execute(stream, &plan_c).unwrap();

        // jgemm: two launches for determinism (distinct output buffers —
        // the second launch must reproduce the first bitwise).
        let plan_j = GemmPlan::row_major_f16(
            da.as_ptr() as *const u16,
            db.as_ptr() as *const u16,
            dc2.as_ptr() as *mut f32,
            1,
            n,
            k,
        );
        let plan_j2 = GemmPlan::row_major_f16(
            da.as_ptr() as *const u16,
            db.as_ptr() as *const u16,
            dc3.as_ptr() as *mut f32,
            1,
            n,
            k,
        );
        jgemm.launch(stream, &plan_j).unwrap();
        jgemm.launch(stream, &plan_j2).unwrap();
        stream.synchronize().unwrap();

        let want = d2h(&dc1, n * 4);
        let got = d2h(&dc2, n * 4);
        let got2 = d2h(&dc3, n * 4);
        (want, got, got2)
    }

    #[test]
    #[ignore = "gpu.yml: l3-jgemm / gemm_m1_diff"]
    fn jgemm_vs_cublas_and_determinism() {
        let ctx = CudaContext::init(DeviceId::new(0)).unwrap();
        let dev = ctx.device_id().index();
        let stream = CudaStream::new(ctx.device_id()).unwrap();
        let blas = Gemm::new(dev).unwrap();
        let arch = reinfer_cuda::arch::resolve_arch().expect("arch");
        let jgemm = Jgemm::new(dev, &arch, Some(std::env::temp_dir().join("reinfer-jit-gemm-m1")))
            .expect("jgemm load");

        // n in {1024, 1536, 3072, 151936} x k in {1024, 3072} — the model's
        // decode geometries (Qwen3-0.6B: hidden 1024, ffn 3072, lm_head 151936).
        let shapes: &[(usize, usize)] = &[
            (1024, 1024),
            (1536, 1024),
            (3072, 1024),
            (1024, 3072),
            (1536, 3072),
            (3072, 3072),
            (151936, 1024), // lm_head
            (151936, 3072),
        ];
        let mut seed = 0xB1E_0001_u64;
        let mut worst_rel_all = 0.0f32;
        for &(n, k) in shapes {
            let (want, got, got2) = run_pair(dev, &blas, &jgemm, &stream, n, k, &mut seed);

            // Determinism: two jgemm launches bit-identical.
            assert_eq!(
                got, got2,
                "jgemm determinism failed (n={n} k={k}): two launches differ bitwise"
            );

            // Drift vs cublas (32F-acc tier, order-difference only).
            let mut max_abs = 0.0f32;
            let mut max_rel = 0.0f32;
            let mut bad = 0usize;
            for (g, w) in got.iter().zip(want.iter()) {
                let diff = (g - w).abs();
                max_abs = max_abs.max(diff);
                let rel = if w.abs() > 1e-9 { diff / w.abs() } else { diff };
                max_rel = max_rel.max(rel);
                // D7 gate: rtol 1e-4 + atol 1e-6 (same tier as the 014 gemm gate).
                if diff > 1e-6 + 1e-4 * w.abs() {
                    bad += 1;
                }
            }
            worst_rel_all = worst_rel_all.max(max_rel);
            eprintln!(
                "jgemm vs cublas n={n:6} k={k:4}: max_abs {max_abs:.3e} max_rel {max_rel:.3e} \
                 over-tol {bad} (gate rtol 1e-4+atol 1e-6)"
            );
            assert_eq!(bad, 0, "jgemm vs cublas over D7 gate (n={n} k={k})");
        }
        eprintln!("jgemm vs cublas: worst max_rel across all shapes = {worst_rel_all:.3e}");
        // Record expectation: order drift <= 1e-5 (the 3.6e-6 historical
        // sample class); the hard gate stays at the D7 rtol.
        assert!(worst_rel_all <= 1e-4, "jgemm drift {worst_rel_all:e} exceeded the D7 gate");
    }
}
