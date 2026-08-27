//! 手动验证：昇腾 L0（specs/011）功能演示——设备 / 流 / 事件 / 缓冲 / 拷贝三链 /
//! 泄漏小压测 / 错误注入。打印每步结果，供人工核对（feature `ffi`，仅目标机运行）。
//!
//! ```text
//! ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest \
//! LD_LIBRARY_PATH=$ASCEND_TOOLKIT_HOME/lib64:$ASCEND_TOOLKIT_HOME/runtime/lib64:$LD_LIBRARY_PATH \
//! cargo run -p reinfer-ascend --features ascend --example basic_ops
//! ```

use reinfer_ascend::{
    AscendContext, AscendDeviceBuffer, AscendEvent, AscendHostBuffer, AscendMemRef, AscendStream,
    copy, copy_async,
};
use reinfer_core::DeviceId;

const ONE_MIB: usize = 1 << 20;

fn ensure_ctx() {
    // 进程内仅一次 aclInit（ACL 初始化/终结非引用计数），泄漏保活至进程结束
    static CTX: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    CTX.get_or_init(|| {
        let ctx = AscendContext::new().expect("aclInit");
        let _ = Box::leak(Box::new(ctx));
    });
    AscendContext::set_device(DeviceId::new(0)).expect("set_device");
}

fn fill_host(buf: &AscendHostBuffer, seed: u8) {
    // SAFETY：pinned host 内存由结构持有，长度固定
    unsafe {
        let s = core::slice::from_raw_parts_mut(buf.as_ptr() as *mut u8, buf.size());
        for (i, b) in s.iter_mut().enumerate() {
            *b = seed.wrapping_add(i as u8);
        }
    }
}

fn xor_checksum(buf: &AscendHostBuffer) -> u64 {
    // SAFETY：只读
    unsafe {
        core::slice::from_raw_parts(buf.as_ptr(), buf.size())
            .iter()
            .fold(0u64, |acc, &b| acc.wrapping_add(acc.wrapping_mul(31).wrapping_add(b as u64)))
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    ensure_ctx();

    println!("== [1] 设备 ==");
    let count = AscendContext::device_count()?;
    println!("device_count = {count}");
    for i in 0..count {
        let info = AscendContext::device_info(i)?;
        println!("  [{i}] soc={}（DeviceProps 待 011 T2）", info.soc_name);
    }

    println!("== [2] 流 / 事件 ==");
    let stream = AscendStream::new()?;
    stream.synchronize()?;
    let evt = AscendEvent::new()?;
    evt.record(Some(&stream))?;
    stream.synchronize()?;
    evt.synchronize()?;
    println!("  record + sync 完成（ACL 事件同步天然阻塞；未 record 行为在探针 P1，此处不演示）");
    drop(evt);

    println!("== [3] 缓冲与拷贝三链（1 MiB 确定性模式）==");
    let src = AscendHostBuffer::alloc(ONE_MIB)?;
    fill_host(&src, 0xA5);
    let d1 = AscendDeviceBuffer::alloc(DeviceId::new(0), ONE_MIB)?;
    let d2 = AscendDeviceBuffer::alloc(DeviceId::new(0), ONE_MIB)?;
    let out_sync = AscendHostBuffer::alloc(ONE_MIB)?;
    copy(&mut AscendMemRef::Device(&d1), &AscendMemRef::Host(&src), ONE_MIB, None)?;
    copy(&mut AscendMemRef::Device(&d2), &AscendMemRef::Device(&d1), ONE_MIB, None)?;
    copy(&mut AscendMemRef::Host(&out_sync), &AscendMemRef::Device(&d2), ONE_MIB, None)?;
    println!(
        "  sync H2D→D2D→D2H checksum src={:016x} out={:016x}",
        xor_checksum(&src),
        xor_checksum(&out_sync)
    );

    let stream2 = AscendStream::new()?;
    let d3 = AscendDeviceBuffer::alloc(DeviceId::new(0), ONE_MIB)?;
    let out_async = AscendHostBuffer::alloc(ONE_MIB)?;
    let e1 =
        copy_async(&mut AscendMemRef::Device(&d3), &AscendMemRef::Host(&src), ONE_MIB, &stream2)?;
    let e2 =
        copy_async(&mut AscendMemRef::Device(&d1), &AscendMemRef::Device(&d3), ONE_MIB, &stream2)?;
    let e3 = copy_async(
        &mut AscendMemRef::Host(&out_async),
        &AscendMemRef::Device(&d1),
        ONE_MIB,
        &stream2,
    )?;
    for e in [&e1, &e2, &e3] {
        e.synchronize()?;
    }
    println!(
        "  async H2D→D2D→D2H checksum src={:016x} out={:016x}",
        xor_checksum(&src),
        xor_checksum(&out_async)
    );

    println!("== [4] 泄漏小压测（100 × 1 MiB alloc/free）==");
    // 无 meminfo 等价暴露（aclrtGetMemInfo 待 cann-rs 0001 增量/T2）——以循环全成功为闸
    for _ in 0..100 {
        let b = AscendDeviceBuffer::alloc(DeviceId::new(0), ONE_MIB)?;
        drop(b);
    }
    println!("  100 次 alloc/free 全部成功（显存前后对比待 T2 追加）");

    println!("== [5] 错误注入演示（精确变体）==");
    let err = AscendDeviceBuffer::alloc(DeviceId::new(0), 1 << 40).expect_err("must fail");
    println!("  over-alloc  -> {err}（预期 Oom）");
    let err = AscendContext::set_device(DeviceId::new(999_999)).expect_err("must fail");
    println!("  bad index   -> {err:?}（预期 Driver/Fatal：非 Oom，fail-closed）");

    println!("OK — 昇腾 L0 功能演示完成");
    Ok(())
}
