//! 独立进程毒化用例（009 T5 注入 (c)，C-F4）。
//!
//! `cudaMemcpyAsync` 以野指针触发 `cudaErrorIllegalAddress(700)` 会**毒化进程内的
//! CUDA 上下文**（后续所有测试连带失败）——因此本文件必须作为独立测试目标存在：
//! 每个 `tests/*.rs` 编译为独立二进制/进程，毒化仅影响本进程。
//!
//! 另外注意：**公开 API 无法构造野指针**（`MemRef`/`DeviceBuffer` 不暴露裸指针、
//! 校验先行）——这正是 009 安全面的证据。本用例只能经 cudarc sys 复核分类映射。

//! ⚠️ 2026-08-27 真机实测：野指针传给 `cudaMemcpyAsync` 在 runtime 层直接
//! **SIGSEGV（驱动不可恢复崩溃）**，不会产生可捕获的 `cudaErrorIllegalAddress`——
//! 因此 009 T5 注入 (c) 的"诱捕方案"废弃（R3 回填）：
//! - 700 → `Driver` 的分类由单元表（error.rs 白名单测试）保证；
//! - 本文件保留为 `#[ignore]` 毒化实验（证明"独立进程隔离"的必要性与
//!   公开 API 无法构造野指针的安全面——`MemRef` 校验证实）。

#[cfg(feature = "cuda")]
mod cases {
    use cudarc::runtime::sys;
    use reinfer_core::DeviceId;
    use reinfer_cuda::{CudaContext, DeviceBuffer};
    use std::ffi::c_void;

    #[test]
    #[ignore = "poison experiment: SIGSEGV (driver crash); classification covered by unit table"]
    fn illegal_address_maps_to_driver_in_isolated_process() {
        let _ctx = CudaContext::init(DeviceId::new(0)).expect("init");
        let src = DeviceBuffer::alloc(DeviceId::new(0), 64).expect("src");
        let dst = 0xDEAD_BEEFu64 as *mut c_void;
        // 本行会 SIGSEGV（实测）——切勿在非隔离进程运行
        let _rc = unsafe {
            sys::cudaMemcpyAsync(
                dst,
                src.as_ptr() as *const c_void,
                64,
                sys::cudaMemcpyKind::cudaMemcpyDeviceToDevice,
                std::ptr::null_mut(),
            )
        };
    }
}
