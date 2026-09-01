//! LogitsView: GPU-resident logits ownership view (006-2 plan D2 L1).
//!
//! The engine owns the device logits buffer and reuses it across decode steps;
//! each step wraps the reused buffer in a fresh `LogitsView` (per-step buffer
//! reuse contract below). This crate is a safe layer: it never dereferences
//! device memory — the backend supplies an opaque handle plus a device-to-host
//! copy closure.
//!
//! Continuity with 014: the host signature `Backend::logits() -> Vec<f32>`
//! (014 plan, Interface Contracts) is inherited by
//! `LogitsView::to_host() -> Vec<f32>` — same element order (row-major
//! `[vocab]`) and same host-facing type; the device-resident view is the
//! 006-2 evolution of that signature (interface patch note, D2 L1).
//!
//! Per-step buffer reuse contract: a view borrows the engine's device buffer
//! for at most one step. `to_host()` is lazy — the device-to-host copy happens
//! on the first call (CPU fallback path only) and is memoized per view. The
//! view must not outlive the step: the engine overwrites the same buffer on
//! the next step, so a view holding a stale host copy must be dropped (a fresh
//! view is created per step). `host_cache` is per view, not per buffer.

use std::fmt;
use std::sync::OnceLock;

use reinfer_core::DeviceId;

/// Opaque handle to device-resident memory.
///
/// The backend (crates/cuda, 006-2 T3D) constructs this from a device pointer
/// value; this crate never dereferences it (`forbid(unsafe_code)`). Only the
/// address value and byte length cross the interface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceBuffer {
    ptr: usize,
    bytes: usize,
}

impl DeviceBuffer {
    /// Wrap a device address (`ptr`) and byte length (`bytes`).
    pub const fn new(ptr: usize, bytes: usize) -> Self {
        Self { ptr, bytes }
    }

    /// Device address value (backend-side consumption point).
    pub const fn ptr(&self) -> usize {
        self.ptr
    }

    /// Byte length of the device buffer.
    pub const fn bytes(&self) -> usize {
        self.bytes
    }
}

/// GPU-resident logits ownership view (006-2 D2 L1).
///
/// `vocab` elements of `f32` row-major; the device buffer layout must satisfy
/// `buffer.bytes() == vocab * size_of::<f32>()` (engine contract, not enforced
/// here). See the module docs for the 014 continuity and per-step reuse
/// contracts.
pub struct LogitsView {
    device: DeviceId,
    buffer: DeviceBuffer,
    vocab: usize,
    copy: Box<dyn Fn() -> Vec<f32> + Send + Sync>,
    host_cache: OnceLock<Vec<f32>>,
}

impl LogitsView {
    /// Wrap a device buffer with its backend-supplied device-to-host copy.
    ///
    /// `copy` is invoked lazily by [`LogitsView::to_host`] (at most once per
    /// view) and must return the logits in the same element order as 014
    /// `Backend::logits()` (row-major `[vocab]`).
    pub fn new(
        device: DeviceId,
        buffer: DeviceBuffer,
        vocab: usize,
        copy: impl Fn() -> Vec<f32> + Send + Sync + 'static,
    ) -> Self {
        Self { device, buffer, vocab, copy: Box::new(copy), host_cache: OnceLock::new() }
    }

    /// Device the logits reside on.
    pub fn device(&self) -> DeviceId {
        self.device
    }

    /// Opaque device buffer handle (GPU sampler path reads device memory).
    pub fn buffer(&self) -> DeviceBuffer {
        self.buffer
    }

    /// Vocabulary size (logits element count).
    pub fn vocab(&self) -> usize {
        self.vocab
    }

    /// Element count (equal to `vocab`).
    pub fn len(&self) -> usize {
        self.vocab
    }

    /// Whether the view holds zero logits.
    pub fn is_empty(&self) -> bool {
        self.vocab == 0
    }

    /// Lazy fallback copy: device-to-host on first call (CPU fallback path),
    /// memoized per view thereafter. Returns an owned `Vec<f32>` with the 014
    /// `Backend::logits()` element order (row-major `[vocab]`). Device copy
    /// errors are the backend's contract at step/launch time (014: logits are
    /// host-visible after the step sync), so this method is infallible.
    pub fn to_host(&self) -> Vec<f32> {
        self.host_cache.get_or_init(|| (self.copy)()).clone()
    }
}

impl fmt::Debug for LogitsView {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LogitsView")
            .field("device", &self.device)
            .field("buffer", &self.buffer)
            .field("vocab", &self.vocab)
            .field("host_cached", &self.host_cache.get().is_some())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)] // 测试断言崩溃即失败
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn to_host_is_lazy_and_memoized() {
        let src: Arc<Vec<f32>> = Arc::new(vec![1.0, 2.0, 3.0, 4.0]);
        let calls = Arc::new(AtomicUsize::new(0));
        let (data, counter) = (src.clone(), calls.clone());
        let view = LogitsView::new(
            DeviceId::new(0),
            DeviceBuffer::new(0x1000, 16),
            4,
            Box::new(move || {
                counter.fetch_add(1, Ordering::SeqCst);
                data.as_ref().clone()
            }),
        );
        assert_eq!(calls.load(Ordering::SeqCst), 0, "copy must be lazy");
        let h1 = view.to_host();
        assert_eq!(h1, *src);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        let h2 = view.to_host();
        assert_eq!(h2, *src);
        assert_eq!(calls.load(Ordering::SeqCst), 1, "host copy memoized per view");
    }

    #[test]
    fn accessors_and_empty_view() {
        let view = LogitsView::new(
            DeviceId::new(3),
            DeviceBuffer::new(0x2000, 8),
            2,
            Box::new(|| vec![1.5, -2.0]),
        );
        assert_eq!(view.device(), DeviceId::new(3));
        assert_eq!(view.buffer(), DeviceBuffer::new(0x2000, 8));
        assert_eq!(view.vocab(), 2);
        assert_eq!(view.len(), 2);
        assert!(!view.is_empty());
        assert_eq!(view.to_host(), vec![1.5, -2.0]);

        let empty = LogitsView::new(
            DeviceId::new(0),
            DeviceBuffer::new(0x3000, 0),
            0,
            Box::new(Vec::<f32>::new),
        );
        assert!(empty.is_empty());
        assert_eq!(empty.to_host(), Vec::<f32>::new());
    }

    #[test]
    fn device_buffer_roundtrip() {
        let b = DeviceBuffer::new(0xABCD, 64);
        assert_eq!(b.ptr(), 0xABCD);
        assert_eq!(b.bytes(), 64);
    }
}
