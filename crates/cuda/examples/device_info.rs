//! 手动验证入口：列出 NVIDIA 设备并演示流/事件链路（feature `cuda`）。
//!
//! ```text
//! cargo run -p reinfer-cuda --features cuda --example device_info
//! ```
//!
//! 预期输出（RTX 5090 判定机）：
//! ```text
//! device_count = 1
//! device[0] NVIDIA GeForce RTX 5090 Laptop GPU cc=12.0 mem=23.42 GiB uuid=0fd2ce94-...
//! event completed = true
//! OK — 设备/流/事件链路验证通过
//! ```

use reinfer_core::DeviceId;
use reinfer_cuda::{CudaContext, CudaEvent, CudaStream};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let count = CudaContext::device_count()?;
    println!("device_count = {count}");

    for i in 0..count {
        let info = CudaContext::device_info(i)?;
        println!(
            "device[{i}] {} cc={}.{} mem={:.2} GiB uuid={}",
            info.name,
            info.major,
            info.minor,
            info.total_mem as f64 / 1024.0 / 1024.0 / 1024.0,
            info.uuid
        );
    }

    // 绑定设备 0（per-thread），演示流与事件链路
    let dev = DeviceId::new(0);
    let _ctx = CudaContext::init(dev)?;
    let stream = CudaStream::new(dev)?;
    stream.synchronize()?;

    let event = CudaEvent::new(dev)?;
    event.record(&stream)?;
    stream.synchronize()?;
    println!("event completed = {}", event.query()?);

    println!("OK — 设备/流/事件链路验证通过");
    Ok(())
}
