//! 昇腾流/事件（011 T3；薄包装 cann 原语，语义差异注记见 011 plan）。

use crate::error::from_cann_error;
use reinfer_kernels::LaunchError;

/// CANN 计算流（`aclrtCreateStream`/`Drop` 销毁；不隐式同步——纪律归上层）。
#[derive(Debug)]
pub struct AscendStream {
    _inner: cann::stream::Stream,
}

impl AscendStream {
    /// 在当前线程绑定设备上创建流。
    pub fn new() -> Result<Self, LaunchError> {
        cann::stream::Stream::new()
            .map(|inner| Self { _inner: inner })
            .map_err(|e| from_cann_error(&e))
    }

    /// 阻塞等待本流全部任务完成。
    pub fn synchronize(&self) -> Result<(), LaunchError> {
        self._inner.synchronize().map_err(|e| from_cann_error(&e))
    }

    /// 查询本流是否空闲（`Ok(true)` = idle）。
    pub fn query(&self) -> Result<bool, LaunchError> {
        self._inner.query().map_err(|e| from_cann_error(&e))
    }

    /// 内部句柄（memcpy_async 需要）。
    pub(crate) fn inner(&self) -> &cann::stream::Stream {
        &self._inner
    }
}

/// CANN 事件（`aclrtCreateEvent`；**同步天然阻塞 CPU**——无 CUDA 侧 BlockingSync 对等物）。
/// 提供 record / synchronize / stream_wait；ACL 无轮询 query（差异注记 011）。
#[derive(Debug)]
pub struct AscendEvent {
    _inner: cann::event::Event,
}

impl AscendEvent {
    /// 创建事件。
    pub fn new() -> Result<Self, LaunchError> {
        cann::event::Event::new()
            .map(|inner| Self { _inner: inner })
            .map_err(|e| from_cann_error(&e))
    }

    /// 在流（None=默认流）上记录事件。
    pub fn record(&self, stream: Option<&AscendStream>) -> Result<(), LaunchError> {
        self._inner.record(stream.map(AscendStream::inner)).map_err(|e| from_cann_error(&e))
    }

    /// 阻塞等待本事件完成（仅本事件；天然线程阻塞）。
    pub fn synchronize(&self) -> Result<(), LaunchError> {
        self._inner.synchronize().map_err(|e| from_cann_error(&e))
    }

    /// 让指定流等待本事件（跨流依赖）。
    pub fn stream_wait(&self, stream: &AscendStream) -> Result<(), LaunchError> {
        self._inner.stream_wait(stream.inner()).map_err(|e| from_cann_error(&e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stub_mode_returns_unavailable() {
        assert!(matches!(AscendStream::new(), Err(LaunchError::Fatal)));
        assert!(matches!(AscendEvent::new(), Err(LaunchError::Fatal)));
    }
}
