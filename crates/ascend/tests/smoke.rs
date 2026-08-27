//! 昇腾 L0 真机冒烟套件（011 T5 验收闸；镜像 crates/cuda/tests/smoke.rs 5 类用例）。
//!
//! 项目规则：本地开发机非 NPU，本 crate **不在本地编译**——本文件只在本套件运行机
//! 上构建。运行（目标机 + 串行）：
//! ```text
//! export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
//! export LD_LIBRARY_PATH=$ASCEND_TOOLKIT_HOME/lib64:$ASCEND_TOOLKIT_HOME/runtime/lib64:$LD_LIBRARY_PATH
//! cargo test -p reinfer-ascend --features ascend --test smoke -- --ignored --test-threads=1
//! ```
//! 与 CUDA 侧差异（镜像对照表/探针清单见 specs/011-ascend-l0-mirror/npu-test-checklist.md）：
//! - `ensure_ctx()` 全局唯一 context（ACL aclInit/aclFinalize 非引用计数——进程内仅一次，
//!   leak 保活至进程结束；重复 aclInit 行为见探针 P2）；
//! - 事件完成态：ACL 同步天然阻塞 CPU，无轮询 query 对等物（cann-rs 未暴露
//!   aclrtQueryEventStatus 系列）——以 record+sync 后 `stream.query() == idle` 校验；
//!   未 record 即同步的行为见探针 P1（timeout 守底）；
//! - 内存总量断言缺口（DeviceProps T2 / aclrtGetMemInfo 待 cann-rs 暴露）：
//!   超量分配用固定保守值 1 TiB；期望分类与实测码见探针 P4；
//! - 非法设备索引的错误码可能 507xxx（recoverable→Driver）或未分类→Fatal，
//!   只拦"非 Oom"；实测变体见探针 P3。

use reinfer_ascend::{
    AscendContext, AscendDeviceBuffer, AscendEvent, AscendHostBuffer, AscendMemRef, AscendStream,
    copy, copy_async,
};
use reinfer_core::DeviceId;
use reinfer_kernels::LaunchError;

const ONE_MIB: usize = 1 << 20;

fn ensure_ctx() {
    static CTX: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    CTX.get_or_init(|| {
        let ctx = AscendContext::new().expect("aclInit");
        // leak：进程内仅一次 aclInit，保活至进程结束（ACL 初始化/终结非引用计数）
        let _ = Box::leak(Box::new(ctx));
    });
    AscendContext::set_device(DeviceId::new(0)).expect("set_device");
}

mod smoke {
    use super::*;

    fn fill_host(buf: &AscendHostBuffer, seed: u8) {
        // SAFETY：pinned host 内存由结构持有，长度固定
        unsafe {
            let s = core::slice::from_raw_parts_mut(buf.as_ptr() as *mut u8, buf.size());
            for (i, b) in s.iter_mut().enumerate() {
                *b = seed.wrapping_add(i as u8);
            }
        }
    }

    fn host_snapshot(buf: &AscendHostBuffer) -> Vec<u8> {
        // SAFETY：同上（只读）
        unsafe { core::slice::from_raw_parts(buf.as_ptr(), buf.size()).to_vec() }
    }

    #[test]
    #[ignore = "npu.yml: smoke"]
    fn device_info_smoke() {
        ensure_ctx();
        let count = AscendContext::device_count().expect("device_count");
        assert!(count >= 1);
        let info = AscendContext::device_info(0).expect("device_info(0)");
        assert!(!info.soc_name.is_empty());
        // 镜像差异：CUDA 侧 name/major/uuid 断言在 DeviceProps（011 T2，cann-rs 增量）
        // 落地后追加；当前 0001 仅有 SoC 名。
    }

    #[test]
    #[ignore = "npu.yml: smoke"]
    fn memcpy_roundtrip() {
        ensure_ctx();
        let src = AscendHostBuffer::alloc(ONE_MIB).expect("src");
        fill_host(&src, 0xA5);
        let d1 = AscendDeviceBuffer::alloc(DeviceId::new(0), ONE_MIB).expect("d1");
        let d2 = AscendDeviceBuffer::alloc(DeviceId::new(0), ONE_MIB).expect("d2");
        let out = AscendHostBuffer::alloc(ONE_MIB).expect("out");

        // 同步三链
        copy(&mut AscendMemRef::Device(&d1), &AscendMemRef::Host(&src), ONE_MIB, None)
            .expect("h2d");
        copy(&mut AscendMemRef::Device(&d2), &AscendMemRef::Device(&d1), ONE_MIB, None)
            .expect("d2d");
        copy(&mut AscendMemRef::Host(&out), &AscendMemRef::Device(&d2), ONE_MIB, None)
            .expect("d2h");
        assert_eq!(host_snapshot(&out), host_snapshot(&src), "sync roundtrip");

        // 异步三链（事件同步凭证）
        let stream = AscendStream::new().expect("stream");
        let e1 =
            copy_async(&mut AscendMemRef::Device(&d1), &AscendMemRef::Host(&src), ONE_MIB, &stream)
                .expect("h2d async");
        let e2 = copy_async(
            &mut AscendMemRef::Device(&d2),
            &AscendMemRef::Device(&d1),
            ONE_MIB,
            &stream,
        )
        .expect("d2d async");
        let e3 =
            copy_async(&mut AscendMemRef::Host(&out), &AscendMemRef::Device(&d2), ONE_MIB, &stream)
                .expect("d2h async");
        for e in [&e1, &e2, &e3] {
            e.synchronize().expect("event sync");
        }
        assert_eq!(host_snapshot(&out), host_snapshot(&src), "async roundtrip");
    }

    #[test]
    #[ignore = "npu.yml: smoke"]
    fn event_query_states() {
        ensure_ctx();
        let stream = AscendStream::new().expect("stream");
        assert!(stream.query().expect("idle before record"), "fresh stream idle");
        let evt = AscendEvent::new().expect("event");
        evt.record(Some(&stream)).expect("record");
        evt.synchronize().expect("event sync (recorded → completed)");
        stream.synchronize().expect("stream sync");
        assert!(stream.query().expect("idle after sync"), "stream query == idle");
        // 镜像差异：ACL 事件同步天然阻塞 CPU；"未 record 即同步"行为见探针 P1。
    }

    #[test]
    #[ignore = "npu.yml: smoke"]
    fn alloc_free_1000_no_leak() {
        ensure_ctx();
        // 镜像差异：无 memGetInfo 等价暴露（aclrtGetMemInfo 待 cann-rs 0001 增量 / T2），
        // 暂以 1000 次 1 MiB alloc/free 全成功为闸——泄漏累计会在第 N 次分配处报错。
        for _ in 0..1000 {
            let buf = AscendDeviceBuffer::alloc(DeviceId::new(0), ONE_MIB).expect("alloc");
            drop(buf);
        }
    }

    #[test]
    #[ignore = "npu.yml: smoke"]
    fn error_injection() {
        ensure_ctx();
        // 1 TiB 固定保守超限（DeviceProps 缺口：暂不能取 total_mem；T2 后改为 total+1，
        // 与 CUDA 侧一致）。期望 207001 段 → Oom；若为 Driver/Fatal 说明 cann-rs
        // is_oom 白名单缺失（回填：cann-rs 补码），见探针 P4。
        let err = AscendDeviceBuffer::alloc(DeviceId::new(0), 1 << 40).expect_err("over-alloc");
        assert!(matches!(err, LaunchError::Oom), "got {err:?}");
        // 非法设备索引：ACL 无 GetDevice 类比，走 set_device 错误路径。码可能 507xxx
        // （recoverable→Driver）或未分类→Fatal；只拦"非 Oom"（三分类均已 fail-closed），
        // 实测变体回填探针 P3。
        let err = AscendContext::set_device(DeviceId::new(999_999)).expect_err("bad dev index");
        assert!(!matches!(err, LaunchError::Oom), "got {err:?}");
    }
}
