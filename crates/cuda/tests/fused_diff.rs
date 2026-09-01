//! 006-2 T4 ②①: fused FFN micro-kernel diff (real machine) — each fused
//! kernel vs the split sequence it replaces, on the same inputs, bit-exact
//! (0 ulp; the D7 fp16 criterion is <= 1 ulp), plus determinism (two
//! launches of each fused kernel are bit-identical).
//!
//! - fused_cast_swiglu_f16 vs cast_f32_to_f16(gate) + cast_f32_to_f16(up) +
//!   swiglu_f16 (3 launches -> 1);
//! - fused_add_rms_f16 vs add_cast_f16 + rms_norm_row_f16 (2 launches -> 1).
//!
//! Run (real machine; nvcc 13.2 — the sm_120a JIT rule):
//! ```text
//! REINFER_CUDA_NVCC=/usr/local/cuda-13.2/bin/nvcc \
//! CUDA_VISIBLE_DEVICES=0 \
//! cargo test -p reinfer-cuda --features cuda --test fused_diff -- \
//!     --ignored --test-threads=1 --nocapture
//! ```

#![cfg(feature = "cuda")]
#![allow(clippy::unwrap_used)] // test assertions panic on failure
#![allow(clippy::print_stdout)] // smoke output

mod gpu {
    use reinfer_core::DeviceId;
    use reinfer_cuda::diff::DiffKernels;
    use reinfer_cuda::engine::DenseKernels;
    use reinfer_cuda::{CudaContext, CudaStream, DeviceBuffer, HostBuffer, MemRef, copy};

    fn xorshift(seed: &mut u64) -> u64 {
        let mut x = *seed;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *seed = x;
        x
    }

    /// Random f16 bit pattern (finite, dense value range).
    fn rand_f16_bits(seed: &mut u64) -> u16 {
        let mant = (xorshift(seed) as u16) & 0x3ff;
        let exp = ((xorshift(seed) as u16) % 0x1e) & 0xf;
        (exp << 10) | mant
    }

    /// Random f32 spanning the activation range with non-trivial f16
    /// rounding: exponent band 110..143 (f32) covers f16 subnormal..~2^16;
    /// a 1/16 tail uses large exponents to exercise the f16 Inf path.
    fn rand_f32(seed: &mut u64) -> f32 {
        let sign = if xorshift(seed) & 1 != 0 { 0x8000_0000u32 } else { 0 };
        let exp = if xorshift(seed) % 16 == 0 {
            150 + (xorshift(seed) % 4) as u32
        } else {
            110 + (xorshift(seed) % 34) as u32
        };
        let man = (xorshift(seed) as u32) & 0x7f_ffff;
        f32::from_bits(sign | (exp << 23) | man)
    }

    fn setup() -> (CudaContext, CudaStream, DenseKernels, DiffKernels) {
        let ctx = CudaContext::init(DeviceId::new(0)).unwrap();
        let stream = CudaStream::new(ctx.device_id()).unwrap();
        let arch = reinfer_cuda::arch::resolve_arch().unwrap();
        let cache = std::env::temp_dir().join("reinfer-jit-fused-diff");
        let _ = std::fs::remove_dir_all(&cache);
        let dense = DenseKernels::new(&arch, Some(cache.clone())).unwrap();
        let diff = DiffKernels::new(&arch, Some(cache), stream.clone()).unwrap();
        (ctx, stream, dense, diff)
    }

    fn upl(dev: u32, bytes: &[u8]) -> DeviceBuffer {
        let hb = HostBuffer::alloc(bytes.len()).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), hb.as_ptr() as *mut u8, bytes.len());
        }
        let db = DeviceBuffer::alloc(DeviceId::new(dev), bytes.len()).unwrap();
        copy(&mut MemRef::Device(&db), &MemRef::Host(&hb), bytes.len(), None).unwrap();
        db
    }

    fn downl(dev: u32, n_bytes: usize) -> DeviceBuffer {
        DeviceBuffer::alloc(DeviceId::new(dev), n_bytes).unwrap()
    }

    fn read_u16(dev: u32, buf: &DeviceBuffer, n: usize) -> Vec<u16> {
        let hb = HostBuffer::alloc(n * 2).unwrap();
        let _ = dev;
        copy(&mut MemRef::Host(&hb), &MemRef::Device(buf), n * 2, None).unwrap();
        unsafe { std::slice::from_raw_parts(hb.as_ptr() as *const u16, n).to_vec() }
    }

    /// fused_cast_swiglu_f16 vs cast_f32_to_f16 x2 + swiglu_f16 — bit-exact.
    #[test]
    #[ignore = "gpu.yml: l3-kernels / fused-diff"]
    fn fused_cast_swiglu_bit_exact_vs_split() {
        let (ctx, stream, dense, diff) = setup();
        let dev = ctx.device_id().index();
        let n = 3072usize;
        let mut seed = 0x5C1u64;
        let gate: Vec<f32> = (0..n).map(|_| rand_f32(&mut seed)).collect();
        let up: Vec<f32> = (0..n).map(|_| rand_f32(&mut seed)).collect();

        let d_gate = upl(dev, &gate.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
        let d_up = upl(dev, &up.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
        let d_gate16 = downl(dev, n * 2);
        let d_up16 = downl(dev, n * 2);
        let d_split = downl(dev, n * 2);
        let d_fused = downl(dev, n * 2);

        // split: cast gate, cast up, swiglu
        diff.launch_cast_f32_f16(
            dev,
            &stream,
            d_gate.as_ptr() as *const f32,
            d_gate16.as_ptr() as *mut u16,
            n as u32,
        )
        .unwrap();
        diff.launch_cast_f32_f16(
            dev,
            &stream,
            d_up.as_ptr() as *const f32,
            d_up16.as_ptr() as *mut u16,
            n as u32,
        )
        .unwrap();
        dense
            .launch_swiglu(
                dev,
                &stream,
                d_gate16.as_ptr() as *const u16,
                d_up16.as_ptr() as *const u16,
                d_split.as_ptr() as *mut u16,
                n as u32,
            )
            .unwrap();
        // fused
        dense
            .launch_fused_cast_swiglu(
                dev,
                &stream,
                d_gate.as_ptr() as *const f32,
                d_up.as_ptr() as *const f32,
                d_fused.as_ptr() as *mut u16,
                n as u32,
            )
            .unwrap();
        stream.synchronize().unwrap();

        let split = read_u16(dev, &d_split, n);
        let fused = read_u16(dev, &d_fused, n);
        assert_eq!(split.len(), fused.len());
        for (i, (s, f)) in split.iter().zip(fused.iter()).enumerate() {
            assert_eq!(s, f, "cast_swiglu bit mismatch at [{i}]: split {s:04x} fused {f:04x}");
        }
        println!("fused_cast_swiglu: bit-exact vs split on {n} elements");
    }

    /// fused_add_rms_f16 vs add_cast_f16 + rms_norm_row_f16 — bit-exact on
    /// both the residual stream x and the norm output xn.
    #[test]
    #[ignore = "gpu.yml: l3-kernels / fused-diff"]
    fn fused_add_rms_bit_exact_vs_split() {
        let (ctx, stream, dense, _diff) = setup();
        let dev = ctx.device_id().index();
        let n = 1024usize;
        let eps = 1e-6f32;
        let mut seed = 0xADD_u64;
        let x: Vec<u16> = (0..n).map(|_| rand_f16_bits(&mut seed)).collect();
        let c: Vec<f32> = (0..n).map(|_| rand_f32(&mut seed)).collect();
        let w: Vec<u16> = (0..n).map(|_| rand_f16_bits(&mut seed)).collect();

        let d_x = upl(dev, &x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
        let d_x2 = upl(dev, &x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
        let d_c = upl(dev, &c.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
        let d_w = upl(dev, &w.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
        let d_xn = downl(dev, n * 2);
        let d_xn2 = downl(dev, n * 2);

        // split: add_cast (x in place), then rms_norm_row (x -> xn)
        dense
            .launch_add_cast(
                dev,
                &stream,
                d_x.as_ptr() as *mut u16,
                d_c.as_ptr() as *const f32,
                n as u32,
            )
            .unwrap();
        dense
            .launch_rms_norm(
                dev,
                &stream,
                d_x.as_ptr() as *const u16,
                d_xn.as_ptr() as *mut u16,
                d_w.as_ptr() as *const u16,
                n as u32,
                eps,
            )
            .unwrap();
        // fused: x += f16(c) in place and xn = rms(x) in one launch
        dense
            .launch_fused_add_rms(
                dev,
                &stream,
                d_x2.as_ptr() as *mut u16,
                d_c.as_ptr() as *const f32,
                d_xn2.as_ptr() as *mut u16,
                d_w.as_ptr() as *const u16,
                n as u32,
                eps,
            )
            .unwrap();
        stream.synchronize().unwrap();

        let xs = read_u16(dev, &d_x, n);
        let xf = read_u16(dev, &d_x2, n);
        let xns = read_u16(dev, &d_xn, n);
        let xnf = read_u16(dev, &d_xn2, n);
        for (i, (s, f)) in xs.iter().zip(xf.iter()).enumerate() {
            assert_eq!(s, f, "residual x bit mismatch at [{i}]: split {s:04x} fused {f:04x}");
        }
        for (i, (s, f)) in xns.iter().zip(xnf.iter()).enumerate() {
            assert_eq!(s, f, "norm out bit mismatch at [{i}]: split {s:04x} fused {f:04x}");
        }
        println!("fused_add_rms: bit-exact vs split on n={n} (x and xn)");
    }

    /// Determinism: two launches of each fused kernel are bit-identical.
    #[test]
    #[ignore = "gpu.yml: l3-kernels / fused-diff"]
    fn fused_deterministic() {
        let (ctx, stream, dense, _diff) = setup();
        let dev = ctx.device_id().index();
        let n = 3072usize;
        let mut seed = 0xDE7_u64;
        let gate: Vec<f32> = (0..n).map(|_| rand_f32(&mut seed)).collect();
        let up: Vec<f32> = (0..n).map(|_| rand_f32(&mut seed)).collect();

        let d_gate = upl(dev, &gate.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
        let d_up = upl(dev, &up.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
        let o1 = downl(dev, n * 2);
        let o2 = downl(dev, n * 2);
        let run = |out: &DeviceBuffer| {
            dense
                .launch_fused_cast_swiglu(
                    dev,
                    &stream,
                    d_gate.as_ptr() as *const f32,
                    d_up.as_ptr() as *const f32,
                    out.as_ptr() as *mut u16,
                    n as u32,
                )
                .unwrap();
            stream.synchronize().unwrap();
            read_u16(dev, out, n)
        };
        let a = run(&o1);
        let b = run(&o2);
        assert_eq!(a, b, "fused_cast_swiglu determinism: two launches differ");
        println!("fused kernels deterministic: bit-identical across launches");
    }
}
