//! L1 真机冒烟套件（009 T6 验收闸；独立测试目标/进程，串行运行）。
//!
//! 运行（008 接线表 `smoke` job / 本地真机）：
//! ```text
//! CUDA_VISIBLE_DEVICES=0 cargo test -p reinfer-cuda --features cuda \
//!     --test smoke -- --ignored --test-threads=1
//! ```
//!
//! 每个用例 `#[ignore]`（→ `--ignored` 才运行），已登记 allowlist
//! （`scripts/ci/ignored-tests.txt`，注释 `gpu.yml: smoke`）。
//!
//! 关系说明：模块内 `ffi_tests` 是开发快速通道（`cargo test --features cuda`），
//! 本套件是 CI/验收通道（隔离进程 + 串行 + 契约可见性）；断言集基本同构。

#[cfg(feature = "cuda")]
mod smoke {
    use reinfer_core::DeviceId;
    use reinfer_cuda::{
        CudaContext, CudaEvent, CudaStream, DeviceBuffer, HostBuffer, buffer::MemRef, copy,
        copy_async,
    };

    const ONE_MIB: usize = 1 << 20;

    fn fill_host(buf: &HostBuffer, seed: u8) {
        // SAFETY：pinned host 内存由结构持有，长度固定
        unsafe {
            let s = core::slice::from_raw_parts_mut(buf.as_ptr() as *mut u8, buf.size());
            for (i, b) in s.iter_mut().enumerate() {
                *b = seed.wrapping_add(i as u8);
            }
        }
    }

    fn host_snapshot(buf: &HostBuffer) -> Vec<u8> {
        // SAFETY：同上（只读）
        unsafe { core::slice::from_raw_parts(buf.as_ptr(), buf.size()).to_vec() }
    }

    #[test]
    #[ignore = "gpu.yml: smoke"]
    fn device_info_smoke() {
        let count = CudaContext::device_count().expect("device_count");
        assert!(count >= 1);
        let ctx = CudaContext::init(DeviceId::new(0)).expect("init dev 0");
        assert_eq!(ctx.device_id(), DeviceId::new(0));
        let info = CudaContext::device_info(0).expect("device_info(0)");
        assert!(!info.name.is_empty());
        assert!(info.major >= 10);
        assert!(info.total_mem > 0);
        let lens: Vec<usize> = info.uuid.split('-').map(|p| p.len()).collect();
        assert_eq!(lens, vec![8, 4, 4, 4, 12]);
        assert_eq!(CudaContext::current_device().expect("current"), DeviceId::new(0));
    }

    #[test]
    #[ignore = "gpu.yml: smoke"]
    fn memcpy_roundtrip() {
        let ctx = CudaContext::init(DeviceId::new(0)).expect("init");
        let dev = ctx.device_id();
        let src = HostBuffer::alloc(ONE_MIB).expect("src");
        fill_host(&src, 0xA5);
        let d1 = DeviceBuffer::alloc(dev, ONE_MIB).expect("d1");
        let d2 = DeviceBuffer::alloc(dev, ONE_MIB).expect("d2");
        let out = HostBuffer::alloc(ONE_MIB).expect("out");

        // 同步三链
        copy(&mut MemRef::Device(&d1), &MemRef::Host(&src), ONE_MIB, None).expect("h2d");
        copy(&mut MemRef::Device(&d2), &MemRef::Device(&d1), ONE_MIB, None).expect("d2d");
        copy(&mut MemRef::Host(&out), &MemRef::Device(&d2), ONE_MIB, None).expect("d2h");
        assert_eq!(host_snapshot(&out), host_snapshot(&src), "sync roundtrip");

        // 异步三链（事件同步凭证）
        let stream = CudaStream::new(dev).expect("stream");
        let e1 = copy_async(&mut MemRef::Device(&d1), &MemRef::Host(&src), ONE_MIB, &stream)
            .expect("h2d async");
        let e2 = copy_async(&mut MemRef::Device(&d2), &MemRef::Device(&d1), ONE_MIB, &stream)
            .expect("d2d async");
        let e3 = copy_async(&mut MemRef::Host(&out), &MemRef::Device(&d2), ONE_MIB, &stream)
            .expect("d2h async");
        for e in [&e1, &e2, &e3] {
            e.synchronize().expect("event sync");
        }
        assert_eq!(host_snapshot(&out), host_snapshot(&src), "async roundtrip");
    }

    #[test]
    #[ignore = "gpu.yml: smoke"]
    fn event_query_states() {
        let ctx = CudaContext::init(DeviceId::new(0)).expect("init");
        let dev = ctx.device_id();
        let stream = CudaStream::new(dev).expect("stream");
        let evt = CudaEvent::new(dev).expect("event");
        assert!(evt.query().expect("unrecorded is completed"), "实测：从未 record 即完成态");
        evt.record(&stream).expect("record");
        stream.synchronize().expect("stream sync");
        assert!(evt.query().expect("completed"));
        drop(evt); // BlockingSync Drop 兜底不挂起
    }

    #[test]
    #[ignore = "gpu.yml: smoke"]
    fn alloc_free_1000_no_leak() {
        use cudarc::runtime::sys;
        let ctx = CudaContext::init(DeviceId::new(0)).expect("init");
        let dev = ctx.device_id();
        let (mut free_before, mut total) = (0usize, 0usize);
        let snapshot = |f: &mut usize, t: &mut usize| {
            unsafe { sys::cudaMemGetInfo(f, t) }.result().expect("mem info");
        };
        snapshot(&mut free_before, &mut total);
        assert!(total > 0);
        for _ in 0..1000 {
            let buf = DeviceBuffer::alloc(dev, ONE_MIB).expect("alloc");
            unsafe {
                sys::cudaMemsetAsync(buf.as_ptr() as *mut _, 0xAB, ONE_MIB, std::ptr::null_mut())
            }
            .result()
            .expect("memset");
            unsafe { sys::cudaStreamSynchronize(std::ptr::null_mut()) }.result().expect("sync");
        }
        let (mut free_after, mut total_after) = (0usize, 0usize);
        snapshot(&mut free_after, &mut total_after);
        assert_eq!(total_after, total);
        let allowance = ONE_MIB * 10 + 8 * ONE_MIB; // 1% 总 + 8 MiB slack
        assert!(free_after >= free_before.saturating_sub(allowance));
    }

    #[test]
    #[ignore = "gpu.yml: smoke"]
    fn error_injection() {
        let ctx = CudaContext::init(DeviceId::new(0)).expect("init");
        let dev = ctx.device_id();
        let total = CudaContext::device_info(0).expect("info").total_mem;
        let err = DeviceBuffer::alloc(dev, total as usize + 1).expect_err("over-alloc");
        assert!(matches!(err, reinfer_kernels::LaunchError::Oom), "got {err:?}");
        let count = CudaContext::device_count().expect("count");
        if count > 0 {
            let err = CudaContext::init(DeviceId::new(count)).expect_err("bad dev index");
            assert!(matches!(err, reinfer_kernels::LaunchError::Fatal), "got {err:?}");
        }
    }
}
