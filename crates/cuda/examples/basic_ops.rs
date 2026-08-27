//! 手动验证：L1（specs/009）全功能演示——设备 / 流 / 事件 / 缓冲 / 拷贝三链 /
//! 泄漏小压测 / 错误注入。打印每步结果，供人工核对。
//!
//! ```text
//! cargo run -p reinfer-cuda --features cuda --example basic_ops
//! ```
#![allow(unsafe_code)]

use reinfer_core::DeviceId;
use reinfer_cuda::{
    CudaContext, CudaEvent, CudaStream, DeviceBuffer, HostBuffer, MemRef, copy, copy_async,
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

fn xor_checksum(buf: &HostBuffer) -> u64 {
    // SAFETY：只读
    unsafe {
        core::slice::from_raw_parts(buf.as_ptr(), buf.size())
            .iter()
            .fold(0u64, |acc, &b| acc.wrapping_add(acc.wrapping_mul(31).wrapping_add(b as u64)))
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("== [1] 设备 ==");
    let count = CudaContext::device_count()?;
    println!("device_count = {count}");
    for i in 0..count {
        let info = CudaContext::device_info(i)?;
        println!(
            "  [{i}] {} cc={}.{} mem={:.2} GiB uuid={}",
            info.name,
            info.major,
            info.minor,
            info.total_mem as f64 / 1024.0 / 1024.0 / 1024.0,
            info.uuid
        );
    }
    let dev = DeviceId::new(0);
    let _ctx = CudaContext::init(dev)?;
    println!("current_device = {}（per-thread 绑定，仅本线程）", CudaContext::current_device()?);

    println!("== [2] 流 / 事件 ==");
    let stream = CudaStream::new(dev)?;
    stream.synchronize()?;
    let evt = CudaEvent::new(dev)?;
    println!("  event(new) query = {}（完成态——未 record 即完成，T2 实测语义）", evt.query()?);
    evt.record(&stream)?;
    stream.synchronize()?;
    println!("  after record + sync query = {}", evt.query()?);
    drop(evt); // BlockingSync Drop 兜底同步 + 销毁（不挂起）
    println!("  event dropped OK");

    println!("== [3] 缓冲与拷贝三链（1 MiB 确定性模式）==");
    let src = HostBuffer::alloc(ONE_MIB)?;
    fill_host(&src, 0xA5);
    let d1 = DeviceBuffer::alloc(dev, ONE_MIB)?;
    let d2 = DeviceBuffer::alloc(dev, ONE_MIB)?;
    let out_sync = HostBuffer::alloc(ONE_MIB)?;
    copy(&mut MemRef::Device(&d1), &MemRef::Host(&src), ONE_MIB, None)?;
    copy(&mut MemRef::Device(&d2), &MemRef::Device(&d1), ONE_MIB, None)?;
    copy(&mut MemRef::Host(&out_sync), &MemRef::Device(&d2), ONE_MIB, None)?;
    println!(
        "  sync H2D→D2D→D2H checksum src={:016x} out={:016x}",
        xor_checksum(&src),
        xor_checksum(&out_sync)
    );

    let stream2 = CudaStream::new(dev)?;
    let d3 = DeviceBuffer::alloc(dev, ONE_MIB)?;
    let out_async = HostBuffer::alloc(ONE_MIB)?;
    let e1 = copy_async(&mut MemRef::Device(&d3), &MemRef::Host(&src), ONE_MIB, &stream2)?;
    let e2 = copy_async(&mut MemRef::Device(&d1), &MemRef::Device(&d3), ONE_MIB, &stream2)?;
    let e3 = copy_async(&mut MemRef::Host(&out_async), &MemRef::Device(&d1), ONE_MIB, &stream2)?;
    for e in [&e1, &e2, &e3] {
        e.synchronize()?;
    }
    println!(
        "  async H2D→D2D→D2H checksum src={:016x} out={:016x}",
        xor_checksum(&src),
        xor_checksum(&out_async)
    );

    println!("== [4] 泄漏小压测（100 × 1 MiB alloc/memset/free）==");
    let (free_before, total) = cudarc_meminfo()?;
    for _ in 0..100 {
        let b = DeviceBuffer::alloc(dev, ONE_MIB)?;
        unsafe {
            cudarc::runtime::sys::cudaMemsetAsync(
                b.as_ptr() as *mut core::ffi::c_void,
                0xAB,
                ONE_MIB,
                core::ptr::null_mut(),
            )
        }
        .result()
        .map_err(cudarc_to_launch)?;
        unsafe { cudarc::runtime::sys::cudaStreamSynchronize(core::ptr::null_mut()) }
            .result()
            .map_err(cudarc_to_launch)?;
    }
    let (free_after, total_after) = cudarc_meminfo()?;
    println!(
        "  free {} GiB → {} GiB（Δ {:+.1} MiB）total 不变={}",
        free_before as f64 / 1024.0 / 1024.0 / 1024.0,
        free_after as f64 / 1024.0 / 1024.0 / 1024.0,
        (free_after as f64 - free_before as f64) / 1024.0 / 1024.0,
        total == total_after
    );

    println!("== [5] 错误注入演示（精确变体）==");
    let total_mem = CudaContext::device_info(0)?.total_mem;
    let err = DeviceBuffer::alloc(dev, total_mem as usize + 1).expect_err("must fail");
    println!("  over-alloc  -> {err}（预期 Oom）");
    let err = CudaContext::init(DeviceId::new(count)).expect_err("must fail");
    println!("  bad index   -> {err}（预期 Fatal：101 不在白名单，fail-closed）");

    println!("OK — L1 全功能演示完成");
    Ok(())
}

fn cudarc_meminfo() -> Result<(usize, usize), Box<dyn std::error::Error>> {
    let (mut free, mut total) = (0usize, 0usize);
    unsafe { cudarc::runtime::sys::cudaMemGetInfo(&mut free, &mut total) }
        .result()
        .map_err(|e| Box::new(LaunchErrorWrapper(e)))?;
    Ok((free, total))
}

fn cudarc_to_launch(e: cudarc::runtime::result::RuntimeError) -> Box<dyn std::error::Error> {
    Box::new(LaunchErrorWrapper(e))
}

/// 包装以支持 `?`（Box<dyn Error> 需要 std::error::Error；RuntimeError 未实现）。
struct LaunchErrorWrapper(cudarc::runtime::result::RuntimeError);
impl std::fmt::Debug for LaunchErrorWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "cudarc runtime error")
    }
}
impl std::fmt::Display for LaunchErrorWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "cuda error code {}", self.0.0 as i32)
    }
}
impl std::error::Error for LaunchErrorWrapper {}
