//! 手动验证入口：列出昇腾设备（feature `ffi`，仅目标机运行）。
//!
//! ```text
//! ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest \
//! LD_LIBRARY_PATH=$ASCEND_TOOLKIT_HOME/lib64:$ASCEND_TOOLKIT_HOME/runtime/lib64:$LD_LIBRARY_PATH \
//! cargo run -p reinfer-ascend --features ascend --example device_info
//! ```
//!
//! 预期输出（Ascend 910B 判定机）：
//! ```text
//! device_count = 1
//! device[0] soc=Ascend910B1
//! OK — 设备信息验证完成
//! ```
//! 注：字段仅 SoC 名（cann-rs 0001 现状；DeviceProps = 011 T2 缺口）。

use reinfer_ascend::AscendContext;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ctx = AscendContext::new()?;
    let count = AscendContext::device_count()?;
    println!("device_count = {count}");

    for i in 0..count {
        let info = AscendContext::device_info(i)?;
        println!("device[{i}] soc={}（DeviceProps 待 011 T2）", info.soc_name);
    }

    println!("OK — 设备信息验证完成");
    Ok(())
}
