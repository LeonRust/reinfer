//! 012 C3：vec_add Jit 链路真机冒烟（差分 / 命中预算 / 确定性）。
//!
//! 运行（任意 N 卡；`--test-threads=1` 强制）。`REINFER_CUDA_ARCH` 可选：
//! 未设置时按设备实测 `sm_{{cc}}`（`arch::resolve_arch`）；`-a` 后缀仅在
//! 需要 arch-specific 特性时显式指认。
//! ```text
//! REINFER_CUDA_NVCC=/usr/local/cuda-12.8/bin/nvcc \
//! REINFER_JIT_CACHE=/tmp/reinfer-jit-smoke \
//! CUDA_VISIBLE_DEVICES=0 \
//! cargo test -p reinfer-cuda --features cuda --test jit_smoke -- \
//!     --ignored --test-threads=1
//! ```

#[cfg(feature = "cuda")]
mod smoke {
    use reinfer_core::{DType, DeviceId};
    use reinfer_cuda::diff::DiffKernels;
    use reinfer_cuda::jit_provider::{VecAddArgs, VecAddProvider};
    use reinfer_cuda::{CudaContext, CudaStream, DeviceBuffer, HostBuffer, MemRef, copy};
    use reinfer_kernels::{KernelProvider, OpConfig};

    const N: u32 = 1 << 20; // 1 Mi 元素（4 MiB/缓冲）

    fn nvcc_arch() -> String {
        // 无默认特判：env 显式覆盖，否则按设备实测（arch::resolve_arch）
        reinfer_cuda::arch::resolve_arch().expect("resolve arch")
    }

    fn host_data(seed: f32) -> (Vec<f32>, Vec<f32>) {
        let a: Vec<f32> = (0..N as usize)
            .map(|i| seed * (i as f32 * 0.001).sin() + (i % 7) as f32 * 0.25)
            .collect();
        let b: Vec<f32> =
            (0..N as usize).map(|i| (i as f32 * 0.0005).cos() - (i % 3) as f32 * 0.5).collect();
        (a, b)
    }

    fn fill(buf: &HostBuffer, data: &[f32]) {
        // SAFETY：pinned host 内存由结构持有；长度为 F32 计数 *4
        unsafe {
            let s = core::slice::from_raw_parts_mut(buf.as_ptr() as *mut f32, data.len());
            s.copy_from_slice(data);
        }
    }

    fn snapshot(buf: &HostBuffer, n: usize) -> Vec<f32> {
        // SAFETY：只读（同上）
        unsafe { core::slice::from_raw_parts(buf.as_ptr() as *const f32, n).to_vec() }
    }

    fn make_provider(ctx: &CudaContext, cache: &std::path::Path) -> VecAddProvider {
        let stream = CudaStream::new(ctx.device_id()).expect("stream");
        VecAddProvider::new(&nvcc_arch(), Some(cache.to_path_buf()), stream).expect("provider")
    }

    #[test]
    #[ignore = "gpu.yml: smoke / l2-jit"]
    fn vec_add_diff_and_determinism() {
        let ctx = CudaContext::init(DeviceId::new(0)).expect("ctx");
        let dev = ctx.device_id();
        let cache = std::env::temp_dir().join("reinfer-jit-c3-diff");
        let _ = std::fs::remove_dir_all(&cache);

        let (a, b) = host_data(0.5);
        let hs = HostBuffer::alloc(N as usize * 4).expect("hs");
        let hb = HostBuffer::alloc(N as usize * 4).expect("hb");
        let hout = HostBuffer::alloc(N as usize * 4).expect("hout");
        fill(&hs, &a);
        fill(&hb, &b);

        let da = DeviceBuffer::alloc(dev, N as usize * 4).expect("da");
        let db = DeviceBuffer::alloc(dev, N as usize * 4).expect("db");
        let dout = DeviceBuffer::alloc(dev, N as usize * 4).expect("dout");
        copy(&mut MemRef::Device(&da), &MemRef::Host(&hs), N as usize * 4, None).expect("h2d a");
        copy(&mut MemRef::Device(&db), &MemRef::Host(&hb), N as usize * 4, None).expect("h2d b");

        let provider = make_provider(&ctx, &cache);
        let cfg = OpConfig {
            op: "vec_add",
            device: dev,
            in_dt: DType::F32,
            out_dt: DType::F32,
            head_dim: 0,
            batch: 1,
            seq: 0,
        };
        assert!(provider.matches(&cfg));
        let args = VecAddArgs {
            a: da.as_ptr().cast::<f32>(),
            b: db.as_ptr().cast::<f32>(),
            out: dout.as_ptr().cast::<f32>() as *mut f32,
            n: N,
        };
        assert!(VecAddProvider::size_check(N, (&da, &db, &dout)));
        let mut args = args;
        eprintln!("[diff] launching kernel...");
        provider.launch(&cfg, &mut args).expect("launch1");
        eprintln!("[diff] kernel launched, syncing...");
        provider.sync_stream().expect("sync1");
        eprintln!("[diff] synced, copying back...");
        copy(&mut MemRef::Host(&hout), &MemRef::Device(&dout), N as usize * 4, None).expect("d2h");

        let got = snapshot(&hout, N as usize);
        let expect: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
        // D7 容差（fp32 rtol/atol 逐项）
        let mut worst = 0.0f32;
        for (g, e) in got.iter().zip(&expect) {
            let d = (g - e).abs();
            let tol = 1e-5 * e.abs().max(f32::EPSILON) + 1e-7;
            assert!(d <= tol, "mismatch at {g} vs {e}");
            worst = worst.max(d);
        }
        eprintln!("vec_add: worst abs err = {worst:e}");

        // 确定性：同 kernel 再跑一次应 bit-exact
        let mut args2 = VecAddArgs {
            a: da.as_ptr().cast::<f32>(),
            b: db.as_ptr().cast::<f32>(),
            out: dout.as_ptr().cast::<f32>() as *mut f32,
            n: N,
        };
        provider.launch(&cfg, &mut args2).expect("launch2");
        provider.sync_stream().expect("sync2");
        let hout2 = HostBuffer::alloc(N as usize * 4).expect("hout2");
        copy(&mut MemRef::Host(&hout2), &MemRef::Device(&dout), N as usize * 4, None)
            .expect("d2h2");
        assert_eq!(snapshot(&hout2, N as usize), got, "determinism mismatch");
    }

    // ---------- D2：diff 内核差分 ----------

    /// 确定性 host LCG（不受设备随机性影响）。
    fn lcg(state: &mut u64) -> f32 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*state >> 33) as f32 / (1u64 << 31) as f32 - 1.0
    }

    fn host_to_dev(buf: &mut DeviceBuffer, data: &[f32]) {
        let h = HostBuffer::alloc(data.len() * 4).expect("host");
        fill(&h, data);
        copy(&mut MemRef::Device(buf), &MemRef::Host(&h), data.len() * 4, None).expect("h2d");
    }

    fn dev_to_host(buf: &DeviceBuffer, n: usize) -> Vec<f32> {
        let h = HostBuffer::alloc(n * 4).expect("hout");
        copy(&mut MemRef::Host(&h), &MemRef::Device(buf), n * 4, None).expect("d2h");
        snapshot(&h, n)
    }

    fn diff_check(got: &[f32], want: &[f32], what: &str) {
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            let d = (g - w).abs();
            let tol = 1e-5 * w.abs().max(f32::EPSILON) + 1e-7;
            assert!(d <= tol, "{what} mismatch[{i}]: {g} vs {w}");
        }
    }

    #[test]
    #[ignore = "gpu.yml: smoke / l2-jit"]
    fn diff_kernels_rms_norm() {
        let ctx = CudaContext::init(DeviceId::new(0)).expect("ctx");
        let dev = ctx.device_id();
        let cache = std::env::temp_dir().join("reinfer-jit-d2-rms");
        let _ = std::fs::remove_dir_all(&cache);
        let stream = CudaStream::new(dev).expect("stream");
        let dk = DiffKernels::new(&nvcc_arch(), Some(cache), stream).expect("DiffKernels");

        for n in [64usize, 96, 128, 160, 256] {
            let mut seed = 0x5eedu64;
            let x: Vec<f32> = (0..n).map(|_| lcg(&mut seed)).collect();
            let w: Vec<f32> = (0..n).map(|_| lcg(&mut seed)).collect();
            let want = reinfer_kernels::refs::rms_norm_ref(&x, &w, 1e-5);

            let mut dx = DeviceBuffer::alloc(dev, n * 4).expect("dx");
            let mut dw = DeviceBuffer::alloc(dev, n * 4).expect("dw");
            let dout = DeviceBuffer::alloc(dev, n * 4).expect("dout");
            host_to_dev(&mut dx, &x);
            host_to_dev(&mut dw, &w);
            dk.launch_rms_norm(
                0,
                dx.as_ptr().cast(),
                dw.as_ptr().cast(),
                dout.as_ptr().cast::<f32>() as *mut f32,
                n as u32,
                1e-5,
            )
            .expect("launch rms");
            dk.sync_stream().expect("sync");
            let got = dev_to_host(&dout, n);
            diff_check(&got, &want, "rms_norm");

            // 确定性：再跑一遍（同输入同产物）bit-exact
            dk.launch_rms_norm(
                0,
                dx.as_ptr().cast(),
                dw.as_ptr().cast(),
                dout.as_ptr().cast::<f32>() as *mut f32,
                n as u32,
                1e-5,
            )
            .expect("launch rms2");
            dk.sync_stream().expect("sync2");
            assert_eq!(dev_to_host(&dout, n), got, "rms_norm determinism");
        }
    }

    #[test]
    #[ignore = "gpu.yml: smoke / l2-jit"]
    fn diff_kernels_rope() {
        let ctx = CudaContext::init(DeviceId::new(0)).expect("ctx");
        let dev = ctx.device_id();
        let cache = std::env::temp_dir().join("reinfer-jit-d2-rope");
        let _ = std::fs::remove_dir_all(&cache);
        let stream = CudaStream::new(dev).expect("stream");
        let dk = DiffKernels::new(&nvcc_arch(), Some(cache), stream).expect("DiffKernels");

        for half in [32usize, 48, 64] {
            let n = 2 * half;
            let mut seed = 0xabcdu64;
            let x: Vec<f32> = (0..n).map(|_| lcg(&mut seed)).collect();
            let want = reinfer_kernels::refs::rope_ref(&x, half, 7, 5000.0);

            let mut dx = DeviceBuffer::alloc(dev, n * 4).expect("dx");
            let dout = DeviceBuffer::alloc(dev, n * 4).expect("dout");
            host_to_dev(&mut dx, &x);
            dk.launch_rope(
                0,
                dx.as_ptr().cast(),
                dout.as_ptr().cast::<f32>() as *mut f32,
                half as u32,
                7,
                5000.0,
            )
            .expect("launch rope");
            dk.sync_stream().expect("sync");
            diff_check(&dev_to_host(&dout, n), &want, "rope");
        }
    }

    #[test]
    #[ignore = "gpu.yml: smoke / l2-jit"]
    fn diff_kernels_softmax() {
        let ctx = CudaContext::init(DeviceId::new(0)).expect("ctx");
        let dev = ctx.device_id();
        let cache = std::env::temp_dir().join("reinfer-jit-d2-sm");
        let _ = std::fs::remove_dir_all(&cache);
        let stream = CudaStream::new(dev).expect("stream");
        let dk = DiffKernels::new(&nvcc_arch(), Some(cache), stream).expect("DiffKernels");

        for n in [64usize, 96, 128, 160] {
            let mut seed = 0x900du64;
            let mut mask = Vec::with_capacity(n);
            let mut x = vec![0.0f32; n];
            for (i, slot) in x.iter_mut().enumerate() {
                let m = (i * 37) % 10 < 6; // 伪随机混合掩码（确定性）
                mask.push(m);
                *slot = if m { lcg(&mut seed) } else { f32::NEG_INFINITY };
            }
            let want = reinfer_kernels::refs::masked_softmax_ref(&x, &mask);

            let mut dx = DeviceBuffer::alloc(dev, n * 4).expect("dx");
            let dout = DeviceBuffer::alloc(dev, n * 4).expect("dout");
            host_to_dev(&mut dx, &x);
            dk.launch_masked_softmax(
                0,
                dx.as_ptr().cast(),
                dout.as_ptr().cast::<f32>() as *mut f32,
                n as u32,
            )
            .expect("launch softmax");
            dk.sync_stream().expect("sync");
            let got = dev_to_host(&dout, n);
            // 掩码一致即匹配：逐项（无效位两侧均 -inf，不比较数值）
            for (g, (w, m)) in got.iter().zip(want.iter().zip(&mask)) {
                if *m {
                    let d = (g - w).abs();
                    let tol = 1e-5 * w.abs().max(f32::EPSILON) + 1e-7;
                    assert!(d <= tol, "softmax mismatch: {g} vs {w}");
                } else {
                    assert_eq!(*g, 0.0, "invalid entry must be 0 (exp(-inf) semantics)");
                }
            }

            // 全 masked 行（全 -inf 输入 → 全 -inf 输出，两侧语义一致）
            let all_inf = vec![f32::NEG_INFINITY; n];
            let mut dm = DeviceBuffer::alloc(dev, n * 4).expect("dm");
            host_to_dev(&mut dm, &all_inf);
            dk.launch_masked_softmax(
                0,
                dm.as_ptr().cast(),
                dout.as_ptr().cast::<f32>() as *mut f32,
                n as u32,
            )
            .expect("launch all-inf");
            dk.sync_stream().expect("sync");
            assert!(dev_to_host(&dout, n).iter().all(|v| *v == 0.0));
        }
    }

    // ---------- D3：GPU softmax → host sampler 组合差分 ----------

    #[test]
    #[ignore = "gpu.yml: smoke / l2-jit"]
    fn gpu_softmax_sampler_chain() {
        use reinfer_kernels::sampler::{SplitMix64, sample_from_probs};

        let ctx = CudaContext::init(DeviceId::new(0)).expect("ctx");
        let dev = ctx.device_id();
        let cache = std::env::temp_dir().join("reinfer-jit-d3-chain");
        let _ = std::fs::remove_dir_all(&cache);
        let stream = CudaStream::new(dev).expect("stream");
        let dk = DiffKernels::new(&nvcc_arch(), Some(cache), stream).expect("DiffKernels");

        // logits（含一个 -inf 掩码位）；概率远离阈值避免边界翻转 flake
        let n = 4usize;
        let logits = [0.9f32, -0.2, 0.0, f32::NEG_INFINITY];
        let mask = [true, true, true, false];
        let p_cpu = reinfer_kernels::refs::masked_softmax_ref(&logits, &mask);

        let mut dx = DeviceBuffer::alloc(dev, n * 4).expect("dx");
        let dout = DeviceBuffer::alloc(dev, n * 4).expect("dout");
        host_to_dev(&mut dx, &logits);
        dk.launch_masked_softmax(
            0,
            dx.as_ptr().cast(),
            dout.as_ptr().cast::<f32>() as *mut f32,
            n as u32,
        )
        .expect("launch softmax");
        dk.sync_stream().expect("sync");
        let p_gpu = dev_to_host(&dout, n);
        assert!((p_gpu[3]).abs() < 1e-12, "invalid entry ~0");

        // 同 seed 双管线（SplitMix64 为 Copy）
        let root = SplitMix64::new(0xdeadbeef);
        let mut r_gpu = root;
        let mut r_cpu = root;
        let mut seq_gpu = Vec::new();
        let mut seq_cpu = Vec::new();
        for _ in 0..64 {
            seq_gpu.push(sample_from_probs(&p_gpu, &mut r_gpu).expect("gpu sample"));
            seq_cpu.push(sample_from_probs(&p_cpu, &mut r_cpu).expect("cpu sample"));
        }
        assert_eq!(seq_gpu, seq_cpu, "combined pipeline determinism");
        assert!(seq_gpu.iter().all(|t| *t < 3), "never samples masked token");
        eprintln!("gpu_softmax_sampler_chain: 64-token sequence stable");
    }

    #[test]
    #[ignore = "gpu.yml: smoke / l2-jit"]
    fn jit_cache_reload_budget() {
        let ctx = CudaContext::init(DeviceId::new(0)).expect("ctx");
        let cache = std::env::temp_dir().join("reinfer-jit-c3-budget");
        let _ = std::fs::remove_dir_all(&cache);

        let t0 = std::time::Instant::now();
        let p1 = make_provider(&ctx, &cache);
        let first = t0.elapsed();
        eprintln!("first provider build: {first:?}");

        let t1 = std::time::Instant::now();
        let _p2 = make_provider(&ctx, &cache);
        let hit = t1.elapsed();
        eprintln!("second provider (cache hit): {hit:?}");
        assert!(hit < std::time::Duration::from_millis(50), "hit took {hit:?}");
        assert_eq!(p1.arch(), nvcc_arch());
    }
}
