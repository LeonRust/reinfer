//! CUDA Graph pool — decode-only, self-contained stage (specs/006 D4).
//!
//! # What this module provides
//!
//! - **Bucketed decode graphs**: one pool of `CUDA Graph + exec` entries per
//!   bucket. Buckets are sized by decode step length (token steps / seq_len):
//!   `8..=128` in steps of 8, `128..=256` in steps of 16 (24 buckets total,
//!   vLLM-measured curve; power-of-two buckets are deliberately forbidden).
//! - **Single shared device memory pool**: capture workspaces are
//!   sub-allocated from one `cudaMalloc`'d pool (bump allocator). Pool size is
//!   measured via `cudaMemGetInfo` before/after allocation and recorded
//!   (`MemoryProfile`); the expected device-memory increase is `<= 8%`.
//! - **Capture serialization**: captures are serialized by a global mutex
//!   (llama.cpp style); `capture_in_progress()` exposes the capture window so
//!   the engine can honor the `--no-overlap` semantics (no concurrent
//!   launches during capture). The env soft-switch `REINFER_GRAPH_NO_OVERLAP`
//!   (default on) is parsed here and stored on the pool.
//! - **Per-step content refresh (Graph V2, production path)**: the CUDA
//!   13.2 runtime deep-copies every kernel node's params at capture — the
//!   `kernelParams` entries are dereferenced ONCE and the values packed
//!   into driver-owned storage (measured on RTX 5090 / CUDA 13.2: a V2
//!   read-back shows the recorded entries pointing into the driver's own
//!   buffer, holding capture-time cell contents). Scalar arguments are
//!   frozen forever; only pointer targets (device memory) and captured
//!   memcpy nodes (the per-step kv-len upload) are re-read at replay. So
//!   the per-step variables — gather token, rope q/k positions, kv_write
//!   phys/off — must be re-baked into the graph every step: `replay`
//!   unconditionally re-refreshes the declared refresh nodes (85 for
//!   Qwen3-0.6B) via the V2 `cudaGraphNodeSetParams` path, which reads
//!   the CURRENT cell contents at set time, then syncs the exec with
//!   `cudaGraphExecUpdate` (pointer-diff). Measured per-step cost:
//!   ~0.02 ms set + ~0.05 ms update — negligible against the step's GPU
//!   time. All other nodes' params are constant and stay baked.
//! - **Pointer-diff refresh (fail-closed safety net)**: `replay` keeps the
//!   staging-diff path for cases where the capture could not use the
//!   cells (cublas declarations with `REINFER_JGEMM=off`, the fake-step
//!   smoke tests). Dirty nodes are refreshed via `cudaGraphNodeSetParams`
//!   (the V2 generic interface), then synced with `cudaGraphExecUpdate`
//!   (pointer-diff only); `cudaGraphExecUpdate` failure (topology/
//!   function/type changed) falls back to destroy + re-instantiate.
//!   - Refresh requires a **CUDA >= 13 runtime**: `cuLibraryLoadData`
//!     kernels record their nodes with a `CUkernel` handle
//!     (`cudaKernelFunctionTypeKernel`), which 13.x runtimes accept in the
//!     V2 union; the legacy setters fail with
//!     `cudaErrorInvalidDeviceFunction` on every runtime, and 12.x runtimes
//!     validate the handle as a `CUfunction` and reject the `CUkernel` (the
//!     set fails permanently → replay fails closed with `eager_fallback`
//!     counted, and the engine runs eager). When the 13.x V2 setter is
//!     unavailable the dirty path fails closed (Err, no launch) — the
//!     engine counts the fallback to eager.
//! - **Runtime counters per bucket** (atomics + aggregate summary):
//!   `graph_replay`, `eager_fallback`, `exec_update_success`, `reinstantiated`,
//!   `padding_ratio` (recorded at capture time).
//!
//! # Capture contract (engine layer)
//!
//! The engine declares the kernel specs for a capture with
//! [`GraphPool::declare_specs`] before calling `capture`, and passes a step
//! closure that launches the single-step decode kernels on the capture
//! stream (the same code path as eager execution):
//!
//! ```text
//! pool.declare_specs(vec![spec0, spec1, ...])?;
//! let exec = pool.capture(&stream, seq_len, &[], |s| { ...launch kernels... })?;
//! let builder = PtrUpdateBuilder::new(&specs); // per exec, addressed by role
//! builder.set_role(0, PtrRole::A, &cell_a);
//! builder.commit(&mut exec, &stream)?;         // per step: refresh + replay
//! ```
//!
//! `capture` keeps its `specs: &[KernelSpec]` parameter for backward
//! compatibility: a non-empty slice wins; otherwise the `declare_specs`
//! declaration is consumed; otherwise the capture fails closed *before the
//! capture window opens* (0 declared specs can never match a captured node
//! count — the current engine passes an empty slice and runs eager while
//! the cublas-node spec blocker stands, bench/notes.md BLOCKER-A).
//!
//! - `specs` is one [`KernelSpec`] per kernel node, in launch order,
//!   declaring the param layout (fixed 8-byte slots, or a GEMM geometry
//!   description), the refreshable pointer slots with roles, the kernel
//!   handle and the launch geometry (`grid`/`block`/`shared`). The CUDA API
//!   cannot report a kernel node's parameter count or, for kernels loaded
//!   via `cuLibraryLoadData` (the JIT path), its launch parameters back to
//!   the caller: the legacy `cudaKernelNodeParams` has no count field,
//!   `cuFuncGetAttribute`'s `NUM_PARAMS` attribute was removed in CUDA 12,
//!   and `cudaGraphKernelNodeGetParams` fails with
//!   `cudaErrorInvalidDeviceFunction` for new-style kernels. So the caller
//!   declares the specs; after capture graph.rs verifies `specs.len()`
//!   equals the number of captured kernel nodes (same-shape guard).
//!   `handle`/geometry are used only to rebuild the params struct on
//!   refresh and must match what the step closure actually launched —
//!   except for [`NodeRole::CublasGemm`] nodes, whose handle/geometry are
//!   *recovered by read-back* after capture (see "Node-parameter
//!   read-back" below).
//! - `replay` requires a `PtrUpdate` for **every declared pointer slot** of
//!   **every** node — full coverage is enforced (fail-closed) and verified.
//!   Kernel argument values are addresses of host-side variables captured
//!   into the graph node, so an unrefreshed slot would read stale host
//!   memory.
//!
//! # Node-parameter read-back
//!
//! After capture, graph.rs reads library-launched (cublas) kernel nodes
//! back from the driver so the engine never has to know their internal
//! handle/geometry (BLOCKER-A step b, specs/006-2 T-305):
//!
//! 1. V2 generic `cudaGraphNodeGetParams` (CUDA >= 13.2 runtime) — tried
//!    first: the kernel member (handle/grid/block/shared) is readable for
//!    every node kind, and the kernelParams entries are read as integers
//!    only (never dereferenced — they are capture-time cell addresses). The
//!    symbol is resolved at runtime via `libloading` (cudarc's own loading
//!    crate) because it only exists in libcudart >= 13.2: an unconditional
//!    link reference would stop the whole binary from loading on older
//!    runtimes (the runtime-adaptive tests rely on 12.x runtimes failing
//!    closed at runtime, not at load time).
//! 2. Legacy driver-level `cuGraphKernelNodeGetParams` (the `_v2` symbol in
//!    the 13.x ABI — cuda.h maps the legacy name to it) — fallback on
//!    runtimes without the V2 symbol (pre-13.2, e.g. the default 12.6
//!    toolkit on the RTX 5090 machine): the kernelParams array *and the
//!    argument values it points to* are owned by the node (valid until node
//!    destruction; cuda.h docs), so m/n/k/ld/alpha and the operand staging
//!    pointers are directly readable. Never reached on 13.x runtimes: there
//!    the call returns success but hands back a non-kernelParams encoding
//!    for library-launched nodes (cublas records `extra`-style data) —
//!    dereferencing it segfaults (verified on RTX 5090 / CUDA 13.2).
//!
//! The read-back fills the cublas nodes' handle/geometry (used to rebuild
//! the params struct on refresh) and, when the legacy read owns the values,
//! seeds the *uncovered* slots with the driver-owned value addresses so a
//! dirty node can still be refreshed with partial pointer coverage. With
//! only the V2 read available (13.x runtime), partial coverage fails
//! refresh closed until the engine's stable-param-cell step (BLOCKER-A step
//! a) moves every slot into declared cells. Reads are capped at
//! [`MAX_READBACK_SLOTS`] (16): the driver exposes no parameter count, and
//! the measured cublas sgemmEx kernelParams array holds exactly 22 entries
//! (slot 23 derefs OOB).
//!
//! # Pointer-refresh semantics
//!
//! A `PtrUpdate.ptr` is the **address of a stable cell** that holds the
//! argument value the kernel should see at launch (e.g. a field of the
//! engine's step-argument struct). `cudaGraphNodeSetParams` copies the
//! staging addresses into the graph node; at launch the driver dereferences them
//! and reads the cell's *current* value. Keeping the same cells across
//! replays means the launch picks up new values without any refresh;
//! pointing at *new* cells makes the slot dirty and triggers the
//! pointer-diff update path.
//!
//! Graph V2 inverts this: the CAPTURE itself records the cell addresses
//! (the engine's decl-driven launch passes `kernelParams[i] = &cell_i`),
//! so the nodes permanently reference the cells — the dirty path never
//! fires, `seed_staging` makes the first replay clean, and replay is a
//! plain launch forever (see "Replay without refresh" above). The refresh
//! path remains as the fail-closed net for captures that could not use the
//! cells.
//!
//! # Thread discipline
//!
//! Capture must run on a thread bound to the device (`CudaContext::init`).
//! Only one capture may be in flight process-wide; other threads must not
//! launch on the capture stream or on other streams during the capture window
//! (`--no-overlap`; enforced cooperatively via `capture_in_progress()`).
//!
//! # Environment
//!
//! - `REINFER_GRAPH_NO_OVERLAP` — capture-period no-overlap switch (default
//!   on; `0`/`false`/`off` disables).
//! - `REINFER_GRAPH_POOL_MB` — shared pool size in MiB (default 64).
//!
//! Requires CUDA >= 12.0 (flags-based `cudaGraphInstantiate`; the V2
//! `cudaGraphNodeSetParams` entry point since 12.2, and the 13.x runtime for
//! `cuLibraryLoadData` kernel refresh — see above).

use crate::buffer::DeviceBuffer;
use crate::error::{LaunchError, from_runtime_error};
use crate::stream::CudaStream;
use cudarc::driver::sys as dsys;
use cudarc::runtime::sys;
use libloading::Library;
use reinfer_core::DeviceId;
use std::ffi::{c_char, c_void};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

/// Map a runtime error to `LaunchError`, printing the raw driver code.
/// (Diagnostics for the engine layer; the code is otherwise hidden by the
/// fail-closed `Fatal` classification.)
fn graph_rc(context: &str, r: sys::cudaError_t) -> LaunchError {
    eprintln!("reinfer-cuda graph: {context} failed, code={}", r as i32);
    from_runtime_error(cudarc::runtime::result::RuntimeError(r))
}

/// Runtime version pair (major, minor) from `cudaRuntimeGetVersion` via
/// the LINKED libcudart symbol — note this is version-pinned to the
/// build-time runtime (the reference is `cudaRuntimeGetVersion@libcudart.so.X`),
/// so LD_PRELOAD of a newer libcudart does NOT change it. For the
/// runtime-adaptive refresh/replay gates use [`v2_params_available`]
/// (dlsym-based — sees the LD_PRELOADed runtime); this stays useful for
/// diagnostics and the read-back-source expectations.
pub fn runtime_version() -> Result<(u32, u32), LaunchError> {
    let mut v: i32 = 0;
    // SAFETY: output slot valid.
    let r = unsafe { sys::cudaRuntimeGetVersion(&mut v) };
    if r != sys::cudaError_t::cudaSuccess {
        return Err(graph_rc("cudaRuntimeGetVersion", r));
    }
    Ok(((v / 1000) as u32, ((v % 1000) / 10) as u32))
}

// ---------------------------------------------------------------------------
// Pure bucket table (unit-testable without a GPU)
// ---------------------------------------------------------------------------

/// Smallest decoded step length that is captured (`8`).
pub const BUCKET_MIN: u32 = 8;
/// Bucket step below the split (`8`).
pub const BUCKET_STEP_SMALL: u32 = 8;
/// Bucket step above the split (`16`).
pub const BUCKET_STEP_LARGE: u32 = 16;
/// Step length at which the bucket step granularity changes (`128`).
pub const BUCKET_SPLIT: u32 = 128;
/// Largest captured step length (`256`; larger steps run eager).
pub const BUCKET_MAX: u32 = 256;
/// Number of `8..=128` buckets (`16`).
pub const BUCKET_SMALL: usize = 16;
/// Number of `144..=256` buckets (`8`).
pub const BUCKET_LARGE: usize = 8;
/// Total bucket count (`24`).
pub const BUCKET_COUNT: usize = BUCKET_SMALL + BUCKET_LARGE;

/// Bucket index for a decode step length. `seq_len < BUCKET_MIN` clamps to
/// bucket 0 (its `padding_ratio` then reflects the over-capture); `seq_len >
/// BUCKET_MAX` clamps to the last bucket.
#[must_use]
pub const fn bucket_index(seq_len: u32) -> usize {
    // Smallest bucket whose capacity is >= seq_len: `i = ceil(seq/8) - 1`.
    // `seq_len < BUCKET_MIN` clamps to bucket 0 (its `padding_ratio` then
    // reflects the over-capture); `seq_len > BUCKET_MAX` clamps to the last
    // bucket.
    if seq_len <= BUCKET_SPLIT {
        let i = seq_len.div_ceil(BUCKET_STEP_SMALL);
        if i == 0 {
            return 0;
        }
        let idx = i as usize - 1;
        if idx >= BUCKET_SMALL { BUCKET_SMALL - 1 } else { idx }
    } else {
        let i = (seq_len - BUCKET_SPLIT).div_ceil(BUCKET_STEP_LARGE);
        let idx = BUCKET_SMALL + i as usize - 1;
        if idx >= BUCKET_COUNT { BUCKET_COUNT - 1 } else { idx }
    }
}

/// Upper bound (capture capacity) of a bucket, in step-length units.
#[must_use]
pub const fn bucket_size(index: usize) -> u32 {
    if index < BUCKET_SMALL {
        BUCKET_MIN + index as u32 * BUCKET_STEP_SMALL
    } else {
        BUCKET_SPLIT + (index as u32 + 1 - BUCKET_SMALL as u32) * BUCKET_STEP_LARGE
    }
}

/// Padding ratio for a step length: `(bucket_capacity - seq_len) / capacity`,
/// i.e. the fraction of the captured graph that is idle at this step length.
/// `seq_len > BUCKET_MAX` reports 0 (clamped to the last bucket).
#[must_use]
pub fn padding_ratio(seq_len: u32) -> f32 {
    let cap = bucket_size(bucket_index(seq_len));
    let active = seq_len.min(cap);
    (cap - active) as f32 / cap as f32
}

/// Align `n` up to a multiple of `align` (`n.div_ceil(align) * align`).
#[must_use]
pub fn align_up(n: usize, align: usize) -> usize {
    n.div_ceil(align) * align
}

// ---------------------------------------------------------------------------
// Environment soft switches (pure parsing, unit-testable)
// ---------------------------------------------------------------------------

/// Env var for the capture-period no-overlap switch (default on).
pub const NO_OVERLAP_ENV: &str = "REINFER_GRAPH_NO_OVERLAP";
/// Env var for the shared pool size in MiB.
pub const POOL_MB_ENV: &str = "REINFER_GRAPH_POOL_MB";
/// Default shared pool size in bytes (64 MiB).
pub const DEFAULT_POOL_MB: usize = 64;
/// Alignment of pool sub-allocations (device pointer alignment requirement).
pub const POOL_ALIGN: usize = 256;

/// Pure parser for `REINFER_GRAPH_NO_OVERLAP`: unset or any value other than
/// `0`/`false`/`off` (case-insensitive) yields `true` (on by default).
#[must_use]
pub fn no_overlap_from_env_value(value: Option<&str>) -> bool {
    match value.map(str::trim) {
        Some(s)
            if s.eq_ignore_ascii_case("0")
                || s.eq_ignore_ascii_case("false")
                || s.eq_ignore_ascii_case("off") =>
        {
            false
        }
        _ => true,
    }
}

/// Pure parser for `REINFER_GRAPH_POOL_MB`: valid positive integer -> bytes;
/// unset/empty/non-numeric/zero -> `None` (caller falls back to
/// [`DEFAULT_POOL_MB`]).
#[must_use]
pub fn pool_size_from_env_value(value: Option<&str>) -> Option<usize> {
    let s = value?.trim();
    if s.is_empty() {
        return None;
    }
    let mb: usize = s.parse().ok()?;
    if mb == 0 { None } else { Some(mb * 1024 * 1024) }
}

// ---------------------------------------------------------------------------
// Counters (feature-independent)
// ---------------------------------------------------------------------------

/// Per-bucket runtime counters (atomics; aggregated across execs of the same
/// bucket). `padding_ratio` is recorded at capture time (f32 bit-cast).
#[derive(Debug, Default)]
pub(crate) struct BucketCounters {
    graph_replay: AtomicU64,
    eager_fallback: AtomicU64,
    exec_update_success: AtomicU64,
    reinstantiated: AtomicU64,
    padding_ratio_bits: AtomicU64,
}

/// Snapshot of [`BucketCounters`] (plain, non-atomic).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CounterSnapshot {
    /// Successful `cudaGraphLaunch` count.
    pub graph_replay: u64,
    /// ExecUpdate failures served via destroy + re-instantiate.
    pub eager_fallback: u64,
    /// Successful exec-level pointer-refresh count.
    pub exec_update_success: u64,
    /// Re-instantiation count (each implies one `eager_fallback`).
    pub reinstantiated: u64,
    /// Padding ratio recorded at capture time (`(capacity - seq_len)/capacity`).
    pub padding_ratio: f32,
}

impl CounterSnapshot {
    /// Zero snapshot.
    #[must_use]
    pub const fn zero() -> Self {
        Self {
            graph_replay: 0,
            eager_fallback: 0,
            exec_update_success: 0,
            reinstantiated: 0,
            padding_ratio: 0.0,
        }
    }
}

impl BucketCounters {
    fn snapshot(&self) -> CounterSnapshot {
        CounterSnapshot {
            graph_replay: self.graph_replay.load(Ordering::Relaxed),
            eager_fallback: self.eager_fallback.load(Ordering::Relaxed),
            exec_update_success: self.exec_update_success.load(Ordering::Relaxed),
            reinstantiated: self.reinstantiated.load(Ordering::Relaxed),
            padding_ratio: f32::from_bits(self.padding_ratio_bits.load(Ordering::Relaxed) as u32),
        }
    }
}

/// Aggregate counter snapshot over all buckets.
#[derive(Debug, Clone, PartialEq)]
pub struct PoolSummary {
    /// Per-bucket snapshots, ordered by bucket index.
    pub per_bucket: Vec<(usize, CounterSnapshot)>,
    /// Sum over all buckets.
    pub total: CounterSnapshot,
}

// ---------------------------------------------------------------------------
// Pointer refresh
// ---------------------------------------------------------------------------

/// One refreshed kernel-argument slot: node `node`, parameter slot `slot`,
/// new value `ptr` — the **address of a stable cell** holding the argument
/// (see module docs "Pointer-refresh semantics"). The address is copied into
/// the graph node at refresh time; the driver reads the cell's current value
/// at every launch.
#[derive(Debug, Clone, Copy)]
pub struct PtrUpdate {
    /// Kernel node index in capture order (`0..exec.node_count()`).
    pub node: usize,
    /// Parameter slot index within the node's kernel (a declared pointer
    /// slot of the node's spec — see `KernelSpec::ptr_slots`).
    pub slot: usize,
    /// Address of the stable cell holding the argument value.
    pub ptr: *mut c_void,
}

/// Role of a captured node: how it is produced and what graph.rs must do to
/// make it refreshable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NodeRole {
    /// cuBLAS GEMM node: launched indirectly by the library on the capture
    /// stream. The engine cannot know the internal kernel's handle or
    /// launch geometry, so graph.rs recovers them by reading the node back
    /// after capture (see "Node-parameter read-back" in the module docs).
    CublasGemm,
    /// JIT kernel node launched via `cuLaunchKernel` with engine-owned
    /// argument cells (C3 discipline). The engine declares handle/geometry
    /// and must declare every slot of the layout as a pointer slot.
    CustomKernel,
    /// memcpy node — reserved. memcpy nodes replay with their capture-time
    /// parameters (stable pinned buffers), so no declaration is needed for
    /// the kernel-node capture guard; refresh support is a later wave.
    /// Declaring a spec with this role fails the capture closed.
    Memcpy,
}

/// Role of a kernel-argument slot: the engine's addressing handle for
/// launch-period pointer refresh.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PtrRole {
    /// GEMM operand A.
    A,
    /// GEMM operand B.
    B,
    /// GEMM output C.
    C,
    /// GEMM scratch/workspace pointer (if the algorithm uses one).
    Workspace,
    /// Any other engine-managed argument cell (custom-kernel pointers,
    /// scalar values kept in stable cells, ...).
    Pointer,
}

/// Kernel parameter layout: the driver's kernelParams array viewed as fixed
/// 8-byte-aligned slots — slot `i` covers bytes `[i*8, (i+1)*8)`, matching
/// the CUDA kernel-argument ABI (one address per argument).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamLayout {
    /// Fixed 8-byte-strided slots.
    Fixed {
        /// Total argument count (arity).
        slots: usize,
    },
    /// GEMM geometry description: the 8-byte slot indices holding the m/n/k
    /// argument values. Interpretation aid for the read-back — the engine
    /// knows the values it launched, the read-back shows where they landed.
    Gemm {
        /// Total argument count (arity), as in [`ParamLayout::Fixed`].
        slots: usize,
        /// Slot index of the `m` (rows) argument.
        m: usize,
        /// Slot index of the `n` (columns) argument.
        n: usize,
        /// Slot index of the `k` (reduction depth) argument.
        k: usize,
    },
}

/// Which driver API produced a [`NodeParamsReadback`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadbackSource {
    /// Legacy driver-level `cuGraphKernelNodeGetParams` (the `_v2` symbol
    /// in the 13.x ABI): the kernelParams array *and the argument values it
    /// points to* are owned by the node (valid until node destruction), so
    /// [`NodeParamsReadback::values`] is populated. Only reached on
    /// runtimes without the V2 symbol (pre-13.2; verified on 12.6) — on
    /// 13.x the call succeeds but returns a non-kernelParams encoding for
    /// library-launched nodes.
    Legacy,
    /// V2 generic `cudaGraphNodeGetParams` (CUDA >= 13.2 runtime, resolved
    /// at runtime): the kernel member (handle/grid/block/shared) is
    /// readable for every node kind; [`NodeParamsReadback::values`] is
    /// empty because the kernelParams entries are capture-time cell
    /// addresses (read as integers only, never dereferenced).
    V2,
}

/// A kernel node's parameters as read back from the driver after capture
/// (see "Node-parameter read-back" in the module docs).
#[derive(Debug, Clone)]
pub struct NodeParamsReadback {
    /// Kernel handle as recorded by the node (a `CUkernel` or `CUfunction`
    /// per `function_type`).
    pub handle: *mut c_void,
    /// Handle discriminator (`cudaKernelFunctionType`: 2 = CUkernel,
    /// 3 = CUfunction).
    pub function_type: u32,
    /// Launch grid dimensions.
    pub grid: sys::dim3,
    /// Launch block dimensions.
    pub block: sys::dim3,
    /// Dynamic shared memory bytes.
    pub shared: u32,
    /// The kernelParams entries in slot order — the addresses of the
    /// argument cells recorded at capture. The array is driver-owned and
    /// the entries are safe to read.
    pub cells: Vec<u64>,
    /// Argument *values* read through `cells` — only for
    /// [`ReadbackSource::Legacy`], where the values are driver-owned and
    /// valid until the node is destroyed. Empty for [`ReadbackSource::V2`]
    /// (its cells point at capture-transient memory).
    pub values: Vec<u64>,
    /// Producing driver API.
    pub source: ReadbackSource,
    /// Kernel name when the driver exposes it (`cuKernelGetName` /
    /// `cuFuncGetName`; best effort).
    pub name: Option<String>,
}

impl NodeParamsReadback {
    /// GEMM geometry (m/n/k) extracted from the slot *values* at the slot
    /// indices declared in `layout` — only meaningful when the driver owns
    /// the values ([`ReadbackSource::Legacy`]).
    #[must_use]
    pub fn gemm_mnk(&self, layout: &ParamLayout) -> Option<(u64, u64, u64)> {
        if self.source != ReadbackSource::Legacy {
            return None;
        }
        match *layout {
            ParamLayout::Gemm { m, n, k, .. } => {
                Some((*self.values.get(m)?, *self.values.get(n)?, *self.values.get(k)?))
            }
            ParamLayout::Fixed { .. } => None,
        }
    }
}

/// Pure validation: every declared pointer slot of every node must be
/// covered exactly once (`coverage` = per-node sorted slot-index sets).
#[must_use]
pub(crate) fn validate_updates(updates: &[PtrUpdate], coverage: &[Vec<usize>]) -> bool {
    let mut seen: Vec<Vec<bool>> = coverage.iter().map(|c| vec![false; c.len()]).collect();
    for u in updates {
        let Some(node) = seen.get_mut(u.node) else { return false };
        let Some(pos) = coverage[u.node].binary_search(&u.slot).ok() else {
            return false;
        };
        if node[pos] {
            return false; // duplicate slot
        }
        node[pos] = true;
    }
    seen.iter().all(|node| node.iter().all(|s| *s))
}

/// Pure spec validation: non-empty; every slot index unique and in range;
/// `CustomKernel` specs must declare every slot as a pointer slot (full
/// refresh coverage); the `Memcpy` role is reserved (rejected).
#[must_use]
pub(crate) fn validate_specs(specs: &[KernelSpec]) -> bool {
    if specs.is_empty() {
        return false;
    }
    for spec in specs {
        if spec.role == NodeRole::Memcpy {
            return false;
        }
        let slots = match spec.layout {
            ParamLayout::Fixed { slots } | ParamLayout::Gemm { slots, .. } => slots,
        };
        let mut seen = vec![false; slots];
        for (idx, _role) in &spec.ptr_slots {
            let Some(bit) = seen.get_mut(*idx) else { return false };
            if *bit {
                return false; // duplicate slot
            }
            *bit = true;
        }
        if spec.role == NodeRole::CustomKernel && !seen.iter().all(|s| *s) {
            return false;
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Memory profile
// ---------------------------------------------------------------------------

/// Measured shared-pool memory accounting (profile method, D4).
#[derive(Debug, Clone, Copy)]
pub struct MemoryProfile {
    /// Requested pool size in bytes (one `cudaMalloc`).
    pub pool_bytes: usize,
    /// Measured free-memory delta across the pool allocation
    /// (`cudaMemGetInfo` before/after).
    pub measured_delta_bytes: u64,
    /// Total device memory at pool creation.
    pub total_device_bytes: u64,
    /// Bytes sub-allocated from the pool so far (workspaces).
    pub workspace_used_bytes: usize,
    /// `pool_bytes / total_device_bytes` (expected <= 0.08).
    pub pool_fraction: f64,
}

/// A sub-allocation from the shared pool. Valid while the owning `GraphPool`
/// is alive (pool memory is never moved or reallocated).
#[derive(Debug, Clone, Copy)]
pub struct Workspace {
    ptr: *mut u8,
    size: usize,
}

// SAFETY: the pool memory is stable for the pool's lifetime (single
// cudaMalloc, no growth); the caller keeps the pool alive (execs hold an
// Arc to it). Concurrent use is serialized by the engine (single stream).
unsafe impl Send for Workspace {}

impl Workspace {
    /// Device pointer of the workspace.
    #[inline]
    #[must_use]
    pub const fn ptr(&self) -> *mut u8 {
        self.ptr
    }

    /// Allocated size in bytes (>= the requested size after alignment).
    #[inline]
    #[must_use]
    pub const fn size(&self) -> usize {
        self.size
    }
}

// ---------------------------------------------------------------------------
// Pool
// ---------------------------------------------------------------------------

/// Global capture serialization (llama.cpp style): at most one capture
/// in-flight process-wide.
static CAPTURE_LOCK: Mutex<()> = Mutex::new(());
/// Capture window flag — other threads must not launch while set
/// (capture-period `--no-overlap`).
static CAPTURE_ACTIVE: AtomicBool = AtomicBool::new(false);

/// True while any capture is in flight (all pools, process-wide).
#[must_use]
pub fn capture_in_progress() -> bool {
    CAPTURE_ACTIVE.load(Ordering::Acquire)
}

/// Per-bucket pool bookkeeping.
#[derive(Debug, Default)]
struct BucketStat {
    counters: BucketCounters,
    workspace_bytes: AtomicU64,
}

/// Pool-internal state (all mutation under `state` lock; device buffers are
/// `Send`, so `PoolInner` is `Send + Sync`).
#[derive(Debug)]
struct PoolState {
    enabled: bool,
    no_overlap: bool,
    pool: Option<DeviceBuffer>,
    pool_used: usize,
    pool_requested: usize,
    mem_free_before: u64,
    mem_free_after: u64,
    mem_total: u64,
    buckets: Vec<BucketStat>,
    /// Spec declaration for the next capture (`declare_specs`); consumed by
    /// the next `capture` when the caller passes an empty slice.
    pending_specs: Option<Vec<KernelSpec>>,
}

#[derive(Debug)]
struct PoolInner {
    dev: DeviceId,
    state: Mutex<PoolState>,
}

/// CUDA Graph pool (decode-only). Clone shares the same pool + counters.
#[derive(Debug, Clone)]
pub struct GraphPool {
    inner: Arc<PoolInner>,
}

impl GraphPool {
    /// Create a pool on `dev`, honoring `REINFER_GRAPH_NO_OVERLAP` and
    /// `REINFER_GRAPH_POOL_MB`.
    #[must_use]
    pub fn new(dev: DeviceId) -> Self {
        let no_overlap = no_overlap_from_env_value(std::env::var(NO_OVERLAP_ENV).ok().as_deref());
        let pool_mb = pool_size_from_env_value(std::env::var(POOL_MB_ENV).ok().as_deref())
            .unwrap_or(DEFAULT_POOL_MB * 1024 * 1024);
        Self::new_with(dev, no_overlap, pool_mb)
    }

    /// Create a pool with explicit settings (tests/injection).
    #[must_use]
    pub fn new_with(dev: DeviceId, no_overlap: bool, pool_bytes: usize) -> Self {
        let state = PoolState {
            enabled: true,
            no_overlap,
            pool: None,
            pool_used: 0,
            pool_requested: pool_bytes,
            mem_free_before: 0,
            mem_free_after: 0,
            mem_total: 0,
            buckets: (0..BUCKET_COUNT).map(|_| BucketStat::default()).collect(),
            pending_specs: None,
        };
        Self { inner: Arc::new(PoolInner { dev, state: Mutex::new(state) }) }
    }

    /// A disabled pool: `capture` and workspace allocation always fail. This
    /// is the slow path used when no GPU is available — the engine then runs
    /// fully eager (no graph mode).
    #[must_use]
    pub fn disabled() -> Self {
        let state = PoolState {
            enabled: false,
            no_overlap: true,
            pool: None,
            pool_used: 0,
            pool_requested: 0,
            mem_free_before: 0,
            mem_free_after: 0,
            mem_total: 0,
            buckets: (0..BUCKET_COUNT).map(|_| BucketStat::default()).collect(),
            pending_specs: None,
        };
        Self { inner: Arc::new(PoolInner { dev: DeviceId::new(0), state: Mutex::new(state) }) }
    }

    /// Whether this pool serves captures (false = disabled slow path).
    #[must_use]
    pub fn enabled(&self) -> bool {
        self.inner.state.lock().unwrap_or_else(|p| p.into_inner()).enabled
    }

    /// Capture-period `--no-overlap` setting of this pool.
    #[must_use]
    pub fn no_overlap(&self) -> bool {
        self.inner.state.lock().unwrap_or_else(|p| p.into_inner()).no_overlap
    }

    /// Device of this pool.
    #[must_use]
    pub fn device(&self) -> DeviceId {
        self.inner.dev
    }

    /// Sub-allocate `size` bytes from the shared pool (bump allocator,
    /// 256-byte aligned). The only legal device allocation during capture;
    /// also usable outside capture. Pool-exhaustion is fail-closed.
    pub fn alloc_workspace(&self, size: usize) -> Result<Workspace, LaunchError> {
        let mut state = self.inner.state.lock().unwrap_or_else(|p| p.into_inner());
        if !state.enabled {
            eprintln!("reinfer-cuda graph: pool disabled (no GPU slow path)");
            return Err(LaunchError::Fatal);
        }
        ensure_pool_locked(&self.inner, &mut state)?;
        let pool_size = state.pool.as_ref().map_or(0, |p| p.size());
        let off = align_up(state.pool_used, POOL_ALIGN);
        if off + size > pool_size {
            eprintln!(
                "reinfer-cuda graph: workspace request {} B exceeds pool ({} B total, {} B used) \
                 - raise REINFER_GRAPH_POOL_MB",
                size, pool_size, state.pool_used
            );
            return Err(LaunchError::Fatal);
        }
        state.pool_used = off + size;
        // SAFETY: offset+size within the pool allocation; pool memory stable.
        let ptr = state
            .pool
            .as_ref()
            .map(|p| unsafe { p.as_ptr().cast_mut().add(off) })
            .ok_or(LaunchError::Fatal)?;
        Ok(Workspace { ptr, size })
    }

    /// Start a capture for the bucket of `seq_len` (decode step length).
    /// Serializes with all other captures process-wide; the returned
    /// [`Capture`] holds the global capture lock until `finish`/drop.
    pub fn begin_capture(
        &self,
        stream: &CudaStream,
        seq_len: u32,
    ) -> Result<Capture<'_>, LaunchError> {
        if !self.enabled() {
            eprintln!("reinfer-cuda graph: pool disabled (no GPU slow path)");
            return Err(LaunchError::Fatal);
        }
        let lock = CAPTURE_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        // Pool must exist before capture begins (cudaMalloc is illegal inside
        // a capture window).
        self.ensure_pool()?;
        // SAFETY: stream is a valid runtime stream; mode ThreadLocal keeps
        // other threads' non-capture work legal (cooperation via
        // capture_in_progress() enforces --no-overlap).
        let r = unsafe {
            sys::cudaStreamBeginCapture(
                stream.handle(),
                sys::cudaStreamCaptureMode::cudaStreamCaptureModeThreadLocal,
            )
        };
        if r != sys::cudaError_t::cudaSuccess {
            return Err(graph_rc("cudaStreamBeginCapture", r));
        }
        let pool_used_before = self.inner.state.lock().unwrap_or_else(|p| p.into_inner()).pool_used;
        CAPTURE_ACTIVE.store(true, Ordering::Release);
        Ok(Capture {
            pool: self,
            stream: stream.clone(),
            seq_len,
            bucket: bucket_index(seq_len),
            pool_used_before,
            graph: std::ptr::null_mut(),
            finished: false,
            _lock: lock,
        })
    }

    /// Declare the kernel specs for the next capture (capture-time
    /// contract, module docs). Called by the engine before `capture`; the
    /// declaration is consumed by the next capture. Empty or invalid
    /// declarations are rejected fail-closed (0 declared specs can never
    /// pass the capture count guard).
    pub fn declare_specs(&self, specs: Vec<KernelSpec>) -> Result<(), LaunchError> {
        if !validate_specs(&specs) {
            eprintln!(
                "reinfer-cuda graph: declare_specs rejected — empty or invalid spec \
                 declaration (role/slot coverage; Memcpy role is reserved)"
            );
            return Err(LaunchError::Fatal);
        }
        let mut state = self.inner.state.lock().unwrap_or_else(|p| p.into_inner());
        state.pending_specs = Some(specs);
        Ok(())
    }

    /// Consume the `declare_specs` declaration (always consumed, whether or
    /// not the capture used it — the declaration is one-shot).
    fn take_pending_specs(&self) -> Option<Vec<KernelSpec>> {
        let mut state = self.inner.state.lock().unwrap_or_else(|p| p.into_inner());
        state.pending_specs.take()
    }

    /// Convenience: `begin_capture` -> run step -> `finish`.
    ///
    /// Spec source: a non-empty `specs` slice wins; otherwise the
    /// `declare_specs` declaration is consumed; otherwise the capture fails
    /// closed *before the capture window opens* (the current engine passes
    /// an empty slice and runs eager — BLOCKER-A).
    pub fn capture(
        &self,
        stream: &CudaStream,
        seq_len: u32,
        specs: &[KernelSpec],
        refresh: &[(usize, usize)],
        step: impl FnOnce(&CudaStream) -> Result<(), LaunchError>,
    ) -> Result<GraphExec, LaunchError> {
        let specs: Vec<KernelSpec> = if !specs.is_empty() {
            specs.to_vec()
        } else if let Some(declared) = self.take_pending_specs() {
            declared
        } else {
            eprintln!(
                "reinfer-cuda graph: capture with 0 declared specs fails closed (BLOCKER-A: \
                 the engine cannot declare cublas gemm node specs yet)"
            );
            return Err(LaunchError::Fatal);
        };
        let cap = self.begin_capture(stream, seq_len)?;
        match cap.run(step) {
            Ok(()) => cap.finish(&specs, refresh),
            Err(e) => {
                eprintln!("reinfer-cuda graph: capture step failed: {e:?}");
                Err(e)
            }
        }
    }

    /// Ensure the shared pool exists (one `cudaMalloc`), recording the
    /// `cudaMemGetInfo` before/after delta.
    fn ensure_pool(&self) -> Result<(), LaunchError> {
        let mut state = self.inner.state.lock().unwrap_or_else(|p| p.into_inner());
        ensure_pool_locked(&self.inner, &mut state)
    }

    /// Measured memory accounting of the shared pool.
    #[must_use]
    pub fn memory_stats(&self) -> MemoryProfile {
        let state = self.inner.state.lock().unwrap_or_else(|p| p.into_inner());
        let pool_bytes = state.pool.as_ref().map_or(0, |p| p.size());
        MemoryProfile {
            pool_bytes,
            measured_delta_bytes: state.mem_free_before.saturating_sub(state.mem_free_after),
            total_device_bytes: state.mem_total,
            workspace_used_bytes: state.pool_used,
            pool_fraction: if state.mem_total > 0 {
                pool_bytes as f64 / state.mem_total as f64
            } else {
                0.0
            },
        }
    }

    /// Counter snapshot for one bucket.
    #[must_use]
    pub fn bucket_counters(&self, bucket: usize) -> CounterSnapshot {
        let state = self.inner.state.lock().unwrap_or_else(|p| p.into_inner());
        if bucket < state.buckets.len() {
            state.buckets[bucket].counters.snapshot()
        } else {
            CounterSnapshot::zero()
        }
    }

    /// Workspace bytes sub-allocated during captures of one bucket.
    #[must_use]
    pub fn bucket_workspace_bytes(&self, bucket: usize) -> u64 {
        let state = self.inner.state.lock().unwrap_or_else(|p| p.into_inner());
        state.buckets.get(bucket).map_or(0, |b| b.workspace_bytes.load(Ordering::Relaxed))
    }

    /// Aggregate counter summary over all buckets.
    #[must_use]
    pub fn summary(&self) -> PoolSummary {
        let state = self.inner.state.lock().unwrap_or_else(|p| p.into_inner());
        let mut total = CounterSnapshot::zero();
        let mut per_bucket = Vec::with_capacity(state.buckets.len());
        for (i, b) in state.buckets.iter().enumerate() {
            let s = b.counters.snapshot();
            total.graph_replay += s.graph_replay;
            total.eager_fallback += s.eager_fallback;
            total.exec_update_success += s.exec_update_success;
            total.reinstantiated += s.reinstantiated;
            per_bucket.push((i, s));
        }
        PoolSummary { per_bucket, total }
    }
}

/// Ensure the pool exists (caller holds the state lock).
fn ensure_pool_locked(inner: &PoolInner, state: &mut PoolState) -> Result<(), LaunchError> {
    if state.pool.is_some() {
        return Ok(());
    }
    // Measure free memory before the allocation.
    let mut free: usize = 0;
    let mut total: usize = 0;
    // SAFETY: output slots valid.
    let r = unsafe { sys::cudaMemGetInfo(&mut free, &mut total) };
    if r != sys::cudaError_t::cudaSuccess {
        return Err(graph_rc("cudaMemGetInfo (pool before)", r));
    }
    state.mem_total = total as u64;
    state.mem_free_before = free as u64;
    let pool = DeviceBuffer::alloc(inner.dev, state.pool_requested).map_err(|e| {
        eprintln!("reinfer-cuda graph: pool cudaMalloc failed: {e:?}");
        e
    })?;
    // SAFETY: output slots valid.
    let r = unsafe { sys::cudaMemGetInfo(&mut free, &mut total) };
    if r != sys::cudaError_t::cudaSuccess {
        return Err(graph_rc("cudaMemGetInfo (pool after)", r));
    }
    state.mem_free_after = free as u64;
    state.pool = Some(pool);
    Ok(())
}

// ---------------------------------------------------------------------------
// Capture window
// ---------------------------------------------------------------------------

/// Per-node capture declaration: the engine states what it launched for
/// each kernel node, in launch order. The CUDA API cannot report a kernel
/// node's parameter count or, for kernels loaded via `cuLibraryLoadData`
/// (the JIT path), its launch parameters back to the caller, so the engine
/// declares them and capture verifies the node count (same-shape guard).
/// `handle`/geometry are needed only to rebuild the params struct on
/// pointer refresh and must match what the step closure actually launched —
/// except for [`NodeRole::CublasGemm`] nodes, whose handle/geometry are
/// recovered by the post-capture read-back (see module docs).
#[derive(Debug, Clone)]
pub struct KernelSpec {
    /// How this node is produced (see [`NodeRole`]).
    pub role: NodeRole,
    /// Param layout — fixed 8-byte-strided slots, or a GEMM geometry
    /// description (see [`ParamLayout`]).
    pub layout: ParamLayout,
    /// Refreshable pointer slots: (slot index, role). For
    /// [`NodeRole::CustomKernel`] nodes this must cover every slot of the
    /// layout (fail-closed: an unrefreshable slot would be launched with a
    /// null cell address). For [`NodeRole::CublasGemm`] nodes partial
    /// coverage is allowed: the uncovered slots keep the capture-time
    /// values owned by the driver (see "Node-parameter read-back").
    pub ptr_slots: Vec<(usize, PtrRole)>,
    /// Kernel handle the node was launched with, as the driver records it:
    /// for `cuLibraryLoadData` kernels this is the `CUkernel` from
    /// `cuLibraryGetKernel` (`cudaKernelFunctionTypeKernel`; the graph node
    /// stores the kernel handle, not the `CUfunction` used for the launch).
    /// Preserved for the V2 exec-level params update on refresh. Ignored
    /// for [`NodeRole::CublasGemm`] nodes (recovered by read-back).
    pub handle: *mut c_void,
    /// Launch grid dimensions.
    pub grid: sys::dim3,
    /// Launch block dimensions.
    pub block: sys::dim3,
    /// Dynamic shared memory bytes.
    pub shared: u32,
}

/// Slot count of a param layout.
#[must_use]
pub(crate) const fn layout_slots(layout: &ParamLayout) -> usize {
    match *layout {
        ParamLayout::Fixed { slots } | ParamLayout::Gemm { slots, .. } => slots,
    }
}

/// In-flight capture. Holds the global capture lock; drop aborts the capture
/// (ends the capture window and discards the partial graph) unless `finish`
/// succeeded. `run` executes the step closure, `finish` ends the capture and
/// produces a [`GraphExec`] that owns the graph.
pub struct Capture<'a> {
    pool: &'a GraphPool,
    stream: CudaStream,
    seq_len: u32,
    bucket: usize,
    pool_used_before: usize,
    /// Graph handle produced by `cudaStreamEndCapture` (owned by the
    /// `GraphExec` returned from `finish`; null until then).
    graph: sys::cudaGraph_t,
    /// True once `finish` succeeded — drop must then not abort the capture
    /// window nor touch the (moved) graph.
    finished: bool,
    _lock: MutexGuard<'static, ()>,
}

impl std::fmt::Debug for Capture<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Capture")
            .field("seq_len", &self.seq_len)
            .field("bucket", &self.bucket)
            .field("stream", &self.stream)
            .finish()
    }
}

impl Capture<'_> {
    /// Run the single-step decode kernel sequence on the capture stream.
    /// Kernels must be launched exactly as in eager execution; their specs
    /// (arity/function/geometry) must match the `specs` passed to `finish`.
    pub fn run(
        &self,
        step: impl FnOnce(&CudaStream) -> Result<(), LaunchError>,
    ) -> Result<(), LaunchError> {
        step(&self.stream)
    }

    /// End the capture, enumerate kernel nodes, instantiate the exec, and
    /// record bucket memory/counter accounting. `specs` must cover the
    /// kernel nodes in launch order (same-shape guard).
    pub fn finish(mut self, specs: &[KernelSpec], refresh: &[(usize, usize)]) -> Result<GraphExec, LaunchError> {
        // SAFETY: capture window active on this stream; output slot valid.
        let mut graph: sys::cudaGraph_t = std::ptr::null_mut();
        let r = unsafe { sys::cudaStreamEndCapture(self.stream.handle(), &mut graph) };
        if r != sys::cudaError_t::cudaSuccess {
            return Err(graph_rc("cudaStreamEndCapture", r));
        }
        self.graph = graph;
        // From here on the graph is ours; every error path below must destroy
        // it explicitly (Drop never frees `graph`, it only aborts the window).

        let nodes = match enumerate_kernel_nodes(graph) {
            Ok(nodes) => nodes,
            Err(e) => {
                // SAFETY: graph owned by this capture.
                let _ = unsafe { sys::cudaGraphDestroy(graph) }.result();
                return Err(e);
            }
        };
        if nodes.is_empty() {
            eprintln!("reinfer-cuda graph: capture produced no kernel nodes");
            // SAFETY: graph owned by this capture.
            let _ = unsafe { sys::cudaGraphDestroy(graph) }.result();
            return Err(LaunchError::Fatal);
        }
        if specs.len() != nodes.len() {
            eprintln!(
                "reinfer-cuda graph: capture shape mismatch — {} kernel nodes captured, \
                 {} specs declared",
                nodes.len(),
                specs.len()
            );
            // SAFETY: graph owned by this capture.
            let _ = unsafe { sys::cudaGraphDestroy(graph) }.result();
            return Err(LaunchError::Fatal);
        }
        if !validate_specs(specs) {
            eprintln!(
                "reinfer-cuda graph: invalid spec declaration (role/slot coverage; \
                 Memcpy role is reserved)"
            );
            // SAFETY: graph owned by this capture.
            let _ = unsafe { sys::cudaGraphDestroy(graph) }.result();
            return Err(LaunchError::Fatal);
        }
        // Recover the parameters of library-launched (cublas) nodes from
        // the driver — the engine cannot declare their handle/geometry
        // (see "Node-parameter read-back"). Read-back failure is
        // fail-closed: such a node could never be refreshed.
        let mut readbacks: Vec<Option<NodeParamsReadback>> = Vec::with_capacity(nodes.len());
        for (node, spec) in nodes.iter().zip(specs.iter()) {
            if spec.role == NodeRole::CublasGemm {
                match read_node_params(*node, layout_slots(&spec.layout)) {
                    Ok(rb) => {
                        if rb.handle.is_null() {
                            eprintln!(
                                "reinfer-cuda graph: cublas node read-back gave a null \
                                       kernel handle"
                            );
                            // SAFETY: graph owned by this capture.
                            let _ = unsafe { sys::cudaGraphDestroy(graph) }.result();
                            return Err(LaunchError::Fatal);
                        }
                        readbacks.push(Some(rb));
                    }
                    Err(e) => {
                        eprintln!("reinfer-cuda graph: cublas node read-back failed: {e:?}");
                        // SAFETY: graph owned by this capture.
                        let _ = unsafe { sys::cudaGraphDestroy(graph) }.result();
                        return Err(e);
                    }
                }
            } else {
                readbacks.push(None);
            }
        }
        let nodes: Vec<KernelNode> = nodes
            .into_iter()
            .zip(specs.iter())
            .zip(readbacks.iter())
            .map(|((node, spec), rb)| KernelNode::from_spec(node, spec, rb.as_ref()))
            .collect();

        // SAFETY: output slot valid; graph valid; flags=0 (synchronous
        // instantiation).
        let mut exec: sys::cudaGraphExec_t = std::ptr::null_mut();
        let r = unsafe { sys::cudaGraphInstantiate(&mut exec, graph, 0) };
        if r != sys::cudaError_t::cudaSuccess {
            // SAFETY: graph owned by this capture.
            let _ = unsafe { sys::cudaGraphDestroy(graph) }.result();
            return Err(graph_rc("cudaGraphInstantiate", r));
        }

        let workspace_delta = {
            let state = self.pool.inner.state.lock().unwrap_or_else(|p| p.into_inner());
            state.pool_used - self.pool_used_before
        };
        {
            let state = self.pool.inner.state.lock().unwrap_or_else(|p| p.into_inner());
            let stat = &state.buckets[self.bucket];
            stat.workspace_bytes.fetch_add(workspace_delta as u64, Ordering::Relaxed);
            stat.counters
                .padding_ratio_bits
                .store(padding_ratio(self.seq_len).to_bits() as u64, Ordering::Relaxed);
        }

        self.finished = true;
        Ok(GraphExec {
            pool: Arc::clone(&self.pool.inner),
            exec,
            graph,
            nodes,
            readbacks,
            refresh: refresh.iter().map(|(n, _)| *n).collect(),
            bucket: self.bucket,
            seq_len: self.seq_len,
            workspace_bytes: workspace_delta as u64,
        })
    }
}

impl Drop for Capture<'_> {
    fn drop(&mut self) {
        CAPTURE_ACTIVE.store(false, Ordering::Release);
        if self.finished {
            return; // graph ownership moved to GraphExec; window already closed
        }
        // Abort the capture window if it is still open (finish never ran or
        // failed mid-way) and discard any partial graph.
        let mut out: sys::cudaGraph_t = std::ptr::null_mut();
        // SAFETY: abort is legal in any capture state; outcome ignored.
        let r = unsafe { sys::cudaStreamEndCapture(self.stream.handle(), &mut out) };
        if r == sys::cudaError_t::cudaSuccess && !out.is_null() {
            // SAFETY: `out` is the aborted capture's graph, owned by us.
            let _ = unsafe { sys::cudaGraphDestroy(out) }.result();
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel node bookkeeping
// ---------------------------------------------------------------------------

/// Per-kernel-node bookkeeping: launch geometry/handle (declared by the
/// engine, or recovered by read-back for cublas nodes) + stable staging
/// array for pointer refresh.
#[derive(Debug)]
struct KernelNode {
    node: sys::cudaGraphNode_t,
    handle: *mut c_void,
    grid: sys::dim3,
    block: sys::dim3,
    shared: u32,
    /// Refreshable slot indices (the spec's declared pointer slots,
    /// sorted) — the replay coverage set.
    ptr_slots: Vec<usize>,
    /// True when every slot of the layout is safe to launch after a
    /// refresh: either fully declared (custom kernels) or seeded from
    /// driver-owned values (cublas + legacy read-back).
    refresh_safe: bool,
    /// Stable host staging array (layout slot count entries); values are
    /// copied into the exec on each refresh. Also serves as the
    /// last-applied baseline.
    staging: Vec<*mut c_void>,
    /// Read-back record (cublas nodes; `None` for custom kernels).
    readback: Option<NodeParamsReadback>,
}

impl KernelNode {
    /// Build the bookkeeping for one enumerated node from the engine's
    /// declared spec plus (for cublas nodes) the driver read-back. Staging
    /// starts all-null for the declared pointer slots, so the first replay
    /// always refreshes them (capture-time parameter addresses are
    /// capture-transient host memory and are deliberately never reused);
    /// uncovered slots of cublas nodes are seeded from the driver-owned
    /// values when the legacy read-back owns them.
    fn from_spec(
        node: sys::cudaGraphNode_t,
        spec: &KernelSpec,
        readback: Option<&NodeParamsReadback>,
    ) -> Self {
        let slots = layout_slots(&spec.layout);
        let mut ptr_slots: Vec<usize> = spec.ptr_slots.iter().map(|(i, _)| *i).collect();
        ptr_slots.sort_unstable();
        ptr_slots.dedup();
        let (handle, grid, block, shared) = match readback {
            Some(rb) => (rb.handle, rb.grid, rb.block, rb.shared),
            None => (spec.handle, spec.grid, spec.block, spec.shared),
        };
        let mut staging: Vec<*mut c_void> = vec![std::ptr::null_mut(); slots];
        let mut refresh_safe = true;
        if let Some(rb) = readback {
            if rb.source == ReadbackSource::Legacy {
                // The driver owns the argument values (valid until node
                // destruction): seed each uncovered slot with the address of
                // its driver-owned value, so replay re-reads the
                // capture-time argument (m/n/k/ld/...) instead of null.
                // The read-back itself is capped (`MAX_READBACK_SLOTS`), so
                // declared slots beyond the read stay null — refresh must
                // then fail closed.
                for (i, cell) in rb.cells.iter().enumerate().take(slots) {
                    if !ptr_slots.binary_search(&i).is_ok() {
                        staging[i] = *cell as *mut c_void;
                    }
                }
                refresh_safe = ptr_slots.len() == slots || slots <= rb.cells.len();
            } else {
                // V2 read-back exposes only capture-time cell addresses
                // (capture-transient for library calls): uncovered slots
                // would be launched null — refresh must fail closed.
                refresh_safe = ptr_slots.len() == slots;
            }
        }
        KernelNode {
            node,
            handle,
            grid,
            block,
            shared,
            ptr_slots,
            refresh_safe,
            staging,
            readback: readback.cloned(),
        }
    }

    /// Rebuild the V2 generic `CUgraphNodeParams` pointing at the staging
    /// array (geometry preserved from the declared spec; only pointer-value
    /// slots ever change). The whole struct is zero-initialized first: the
    /// driver requires all reserved bytes to be zero.
    ///
    /// The kernel-member bytes are written at fixed offsets rather than via
    /// cudarc's `CUDA_KERNEL_NODE_PARAMS_v3` fields because that binding
    /// layout is CUDA-version-dependent (12.x: fields up to `extra`; 13.x:
    /// a `func/kern/cuFunc` union plus appended `ctx`/`functionType`), while
    /// the driver-side ABI is stable across 12.2..13.x for the first 56
    /// bytes, with `ctx` (offset 56) and `functionType` (offset 64)
    /// appended on 13.x. The `functionType` discriminator must match the
    /// node's recorded type (see `FN_TYPE`); `ctx` = NULL means "current
    /// context".
    fn make_v2_params(&mut self) -> sys::cudaGraphNodeParams {
        let mut np: sys::cudaGraphNodeParams = unsafe { std::mem::zeroed() };
        np.type_ = sys::cudaGraphNodeType::cudaGraphNodeTypeKernel;
        // SAFETY: the union member is written byte-by-byte right after
        // zero-initialization. `reserved1` aliases the union in every cudarc
        // layout and is at least 232 bytes; all writes below stay inside the
        // kernel member's 72 bytes (offsets per the official
        // `cudaKernelNodeParamsV2` ABI).
        unsafe {
            let k = np.__bindgen_anon_1.reserved1.as_mut_ptr().cast::<u8>();
            let put_u32 = |k: *mut u8, off: usize, v: u32| {
                std::ptr::write_unaligned(k.add(off).cast::<u32>(), v);
            };
            let put_ptr = |k: *mut u8, off: usize, v: *mut c_void| {
                std::ptr::write_unaligned(k.add(off).cast::<*mut c_void>(), v);
            };
            // handle/grid/block/shared/kernelParams/extra (12.x + 13.x).
            put_ptr(k, 0, self.handle);
            put_u32(k, 8, self.grid.x);
            put_u32(k, 12, self.grid.y);
            put_u32(k, 16, self.grid.z);
            put_u32(k, 20, self.block.x);
            put_u32(k, 24, self.block.y);
            put_u32(k, 28, self.block.z);
            put_u32(k, 32, self.shared);
            put_ptr(k, 40, self.staging.as_mut_ptr() as *mut c_void);
            put_ptr(k, 48, std::ptr::null_mut()); // extra: never used
            // 13.x tail: ctx = NULL (current context) and the handle
            // discriminator (see `make_v2_params` docs).
            put_ptr(k, 56, std::ptr::null_mut());
            put_u32(k, 64, FN_TYPE);
        }
        np
    }
}

/// Handle discriminator for the V2 kernel-member union
/// (`cudaKernelFunctionType`), written at byte offset 64 of the kernel
/// member. 2 = `cudaKernelFunctionTypeKernel`: the handle is a
/// `cudaKernel_t` (the `CUkernel` from `cuLibraryGetKernel`). Verified on an
/// RTX 5090 / CUDA 13.2: a capture launched through
/// `cuLaunchKernel(CUfunction)` records the node with
/// `functionType = cudaKernelFunctionTypeKernel`, and the runtime validates
/// the discriminator against the node's recorded type (0 →
/// `cudaErrorInvalidDeviceFunction`; 3 (CUfunction) →
/// `cudaErrorInvalidValue`). CUDA 13.x runtimes accept the `CUkernel` +
/// type-2 pair in `cudaGraphNodeSetParams`; 12.x runtimes validate the
/// handle as a `CUfunction` and reject it (the refresh then fails closed —
/// see the module docs).
const FN_TYPE: u32 = 2; // cudaKernelFunctionTypeKernel

/// `cudaKernelFunctionTypeFunction` (3) — a node recorded with a
/// `CUfunction` handle. Only such nodes are readable through the legacy
/// `cuGraphKernelNodeGetParams` (see `read_node_params`).
const FN_TYPE_FUNCTION: u32 = 3;

/// Upper bound for read-back slot reads: the driver exposes no parameter
/// count, so reads stop at the declared slot count, capped here.
///
/// Measured on this machine (RTX 5090, cublas sgemmEx on
/// `gemm_f32acc(32,64,128)` / `(16,48,96)`): the node's kernelParams array
/// has exactly 22 driver-owned entries — reads of slots 0..=22 succeed,
/// slot 23 segfaults. The cap must stay below the *minimum* arity of any
/// cublas kernel the engine declares for; 16 covers every slot of interest
/// for the measured layout (geometry at 0/1/2, ld block, operand staging
/// pointers at 13/14, alpha at 15) with margin. The engine wave must keep
/// its cublas slot declarations <= this cap.
const MAX_READBACK_SLOTS: usize = 16;

/// Read slot `i` of a kernelParams array as the *address of the argument
/// cell* (the entry value). SAFETY: `array` must be a driver-owned array
/// with at least `i + 1` entries (both read-back paths guarantee this).
///
/// # Safety
/// Caller guarantees the driver-owned array bounds as above.
unsafe fn read_slot_cell(array: *mut *mut c_void, i: usize) -> u64 {
    // SAFETY: array bound per the caller contract; the entry is an address
    // (read as an integer — no dereference).
    unsafe { (*array.add(i)) as u64 }
}

/// Read slot `i` of a kernelParams array as the *argument value* (one
/// dereference). Only safe when the driver owns the values (legacy read:
/// "the argument values it points to are owned by the node"). Kernel
/// arguments are packed into 8-byte slots by the launch ABI, so an 8-byte
/// read stays inside the driver's value buffer.
///
/// # Safety
/// Caller guarantees `array` points at a driver-owned array whose values
/// are owned by the node and readable as 8-byte slots.
unsafe fn read_slot_value(array: *mut *mut c_void, i: usize) -> u64 {
    // SAFETY: array bound and value ownership per the caller contract.
    unsafe { std::ptr::read_unaligned((*array.add(i)).cast::<u64>()) }
}

/// Best-effort kernel-name lookup for a read-back handle; the handle
/// discriminator (`cudaKernelFunctionType`) picks the API: CUkernel vs
/// CUfunction.
fn kernel_name(handle: *mut c_void, function_type: u32) -> Option<String> {
    let mut name_ptr: *const c_char = std::ptr::null();
    // SAFETY: handle as recorded by the node; output slot valid; the name
    // pointer is driver-owned and read immediately below.
    let r = if function_type == FN_TYPE_FUNCTION {
        unsafe { dsys::cuFuncGetName(&mut name_ptr, handle as dsys::CUfunction) }
    } else {
        unsafe { dsys::cuKernelGetName(&mut name_ptr, handle as dsys::CUkernel) }
    };
    if r != dsys::CUresult::CUDA_SUCCESS || name_ptr.is_null() {
        return None;
    }
    // SAFETY: NUL-terminated driver-owned string.
    Some(unsafe { std::ffi::CStr::from_ptr(name_ptr) }.to_string_lossy().into_owned())
}

/// Resolve the V2 generic node read-back symbol (`cudaGraphNodeGetParams`,
/// added in CUDA 13.2) from the already-loaded runtime library, once per
/// process. Resolved at runtime via `libloading` (cudarc's own loading
/// crate) rather than linked, so a build linked against a 13.2 runtime
/// keeps loading on older runtimes — the runtime-adaptive tests rely on
/// 12.x runtimes failing closed at runtime, not at load time.
///
/// The symbol is only accepted when the process's *global runtime scope*
/// is >= 13.2 (`cudaRuntimeGetVersion` via `Library::this()` — the scope
/// the V2 call's own internal references resolve through): calling the
/// symbol from a separately-dlopened libcudart inside a 12.x-scope
/// process segfaults at the call site (verified on this machine — see
/// `v2_node_get_params`).
fn v2_node_get_params() -> Option<
    unsafe extern "C" fn(sys::cudaGraphNode_t, *mut sys::cudaGraphNodeParams) -> sys::cudaError_t,
> {
    type V2Fn = unsafe extern "C" fn(
        sys::cudaGraphNode_t,
        *mut sys::cudaGraphNodeParams,
    ) -> sys::cudaError_t;
    static RESOLVED: OnceLock<Option<V2Fn>> = OnceLock::new();
    *RESOLVED.get_or_init(|| {
        // The V2 function is only callable when the *process's global
        // runtime scope* is >= 13.2 — i.e. when libcudart.so.13 is the
        // first runtime library in the global symbol scope (linked or
        // LD_PRELOADed). Otherwise a second, separately-dlopened libcudart
        // may host the symbol while the 12.x library owns the process
        // state, and the 13.2 function's internal symbol references then
        // resolve against the 12.x library — `cudaGraphNodeGetParams`
        // segfaults at its entry in that state (verified: RTX 5090 /
        // driver 595.84 / 12.6-linked binary dlopening libcudart.so.13,
        // first V2 get crashed inside libcudart.so.13; with libcudart.so.13
        // LD_PRELOADed the same call works). `Library::this()` resolves
        // through the main program's global scope — the same scope the
        // V2 call's internal dependencies use — so its
        // `cudaRuntimeGetVersion` is the discriminator.
        let this = unsafe { libloading::os::unix::Library::this() };
        let ver = unsafe {
            this.get::<libloading::Symbol<unsafe extern "C" fn(*mut i32) -> sys::cudaError_t>>(
                b"cudaRuntimeGetVersion\0",
            )
        }
        .ok()?;
        let mut v: i32 = 0;
        // SAFETY: output slot valid; the global-scope runtime query.
        if unsafe { ver(&mut v) } != sys::cudaError_t::cudaSuccess || v < 13020 {
            return None; // global runtime scope < 13.2 — V2 unusable (legacy path)
        }
        // The runtime library is already mapped into the process by the
        // cudarc dynamic-linking link step; try its soname variants.
        for name in ["libcudart.so", "libcudart.so.13", "libcudart.so.13.2"] {
            // SAFETY: dlopen-only; the handle is dropped immediately (the
            // library stays mapped via the linker reference).
            let Ok(lib) = (unsafe { Library::new(name) }) else { continue };
            // SAFETY: lookup-only; the resolved fn pointer is stored
            // 'static and stays valid for the process lifetime.
            if let Ok(sym) = unsafe { lib.get::<V2Fn>(b"cudaGraphNodeGetParams\0") } {
                return Some(*sym);
            }
        }
        None
    })
}

/// V2 generic `cudaGraphNodeSetParams` resolved at runtime — the refresh
/// path's counterpart of `v2_node_get_params`, with the SAME global
/// runtime-scope gate (the setter's internal references resolve through
/// the global scope, so calling a separately-dlopened libcudart inside a
/// 12.x-scope process would crash at the call site — same verified
/// failure mode as the read).
///
/// Why dlsym and not the linked symbol: the link step pins a VERSIONED
/// reference (`cudaGraphNodeSetParams@libcudart.so.12` on this machine's
/// binaries), and symbol versioning resolves versioned references to the
/// build-time libcudart even when libcudart.so.13 is LD_PRELOADed into
/// the global scope (verified: 12.6-linked binary + LD_PRELOAD
/// libcudart.so.13.2.51 still calls the 12.6 implementation, which
/// rejects `cuLibraryLoadData` CUkernel handles with
/// cudaErrorInvalidValue). The dlsym-resolved symbol bypasses versioning
/// — the first global-scope definition wins, i.e. the preloaded 13.2
/// implementation — and its own internal references resolve through the
/// same 13.2 scope, exactly like the working V2 read path.
fn v2_node_set_params() -> Option<
    unsafe extern "C" fn(sys::cudaGraphNode_t, *mut sys::cudaGraphNodeParams) -> sys::cudaError_t,
> {
    type V2Fn = unsafe extern "C" fn(
        sys::cudaGraphNode_t,
        *mut sys::cudaGraphNodeParams,
    ) -> sys::cudaError_t;
    static RESOLVED: OnceLock<Option<V2Fn>> = OnceLock::new();
    *RESOLVED.get_or_init(|| {
        // Same global-scope discriminator as the read path: the V2
        // function is only callable when libcudart.so.13 is the first
        // runtime library in the global symbol scope (linked or
        // LD_PRELOADed).
        let this = unsafe { libloading::os::unix::Library::this() };
        let ver = unsafe {
            this.get::<libloading::Symbol<unsafe extern "C" fn(*mut i32) -> sys::cudaError_t>>(
                b"cudaRuntimeGetVersion\0",
            )
        }
        .ok()?;
        let mut v: i32 = 0;
        // SAFETY: output slot valid; the global-scope runtime query.
        if unsafe { ver(&mut v) } != sys::cudaError_t::cudaSuccess || v < 13020 {
            return None; // global runtime scope < 13.2 — linked 12.x setter fails closed
        }
        for name in ["libcudart.so", "libcudart.so.13", "libcudart.so.13.2"] {
            // SAFETY: dlopen-only; the handle is dropped immediately (the
            // library stays mapped via the linker reference).
            let Ok(lib) = (unsafe { Library::new(name) }) else { continue };
            // SAFETY: lookup-only; the resolved fn pointer is stored
            // 'static and stays valid for the process lifetime.
            if let Ok(sym) = unsafe { lib.get::<V2Fn>(b"cudaGraphNodeSetParams\0") } {
                return Some(*sym);
            }
        }
        None
    })
}

/// Whether the V2 generic node-params API (read AND set) is usable in
/// this process — i.e. whether the global runtime scope is >= 13.2
/// (libcudart.so.13 linked or LD_PRELOADed; see `v2_node_get_params`).
/// Integration tests gate the replay expectation on this instead of the
/// linked (versioned) `cudaRuntimeGetVersion`, which always reports the
/// build-time runtime regardless of LD_PRELOAD.
#[must_use]
pub fn v2_params_available() -> bool {
    v2_node_set_params().is_some()
}

/// Driver version major (13 for the 595.84 / CUDA 13.2 driver — the
/// 13.x ABI encoding for `cuDriverGetVersion`), `None` when the query
/// fails (treated as fail-open by the caller, which then relies on the
/// legacy read itself). The legacy node read-back safety depends on this
/// (see `read_node_params`).
fn driver_version_major() -> Option<u32> {
    let mut v: i32 = 0;
    // SAFETY: output slot valid; driver API query.
    let r = unsafe { dsys::cuDriverGetVersion(&mut v) };
    if r != dsys::CUresult::CUDA_SUCCESS {
        return None;
    }
    Some((v / 1000) as u32)
}

/// Read a kernel node's parameters back from the driver after capture (see
/// "Node-parameter read-back" in the module docs).
///
/// Tries the V2 generic read first when the runtime exposes it (CUDA >= 13.2,
/// resolved at runtime) — the only read that is safe for *every* node kind:
/// the kernel member (handle/grid/block/shared) is returned directly and the
/// kernelParams entries are read as integers only (never dereferenced).
///
/// Falls back to the legacy driver-level read — on runtimes without the V2
/// symbol this is the only source that yields argument *values* (driver
/// owned), verified working on a 12.6 runtime + pre-13.0 driver. On 13.x
/// drivers the legacy call returns success but hands back a
/// non-kernelParams encoding for library-launched nodes (cublas launches
/// record `extra`-style data), dereferencing it segfaults (verified on
/// RTX 5090 / CUDA 13.2) — the driver-version guard below fails closed on
/// that combination before the deref.
///
/// Both reads failing is an error: the node is not introspectable. Reads up
/// to `slots` (capped at [`MAX_READBACK_SLOTS`]) entries; the driver
/// exposes no parameter count.
fn read_node_params(
    node: sys::cudaGraphNode_t,
    slots: usize,
) -> Result<NodeParamsReadback, LaunchError> {
    let slots = slots.min(MAX_READBACK_SLOTS);
    // 1) V2 generic read (CUDA >= 13.2 runtime, resolved at runtime).
    if let Some(get) = v2_node_get_params() {
        let mut np: sys::cudaGraphNodeParams = unsafe { std::mem::zeroed() };
        // SAFETY: node from our graph; params zeroed (the driver validates
        // the type discriminator against the node).
        let r = unsafe { get(node, &mut np) };
        if r == sys::cudaError_t::cudaSuccess {
            // Extract the kernel member at the fixed ABI offsets (same
            // layout as `make_v2_params`: handle@0, grid@8, block@20,
            // shared@32, kernelParams@40, extra@48, ctx@56, functionType@64).
            // SAFETY: the union member is read byte-by-byte; `reserved1`
            // aliases it in every cudarc layout and is at least 232 bytes;
            // the kernel member is 72 bytes (reads stay inside).
            unsafe {
                let k = np.__bindgen_anon_1.reserved1.as_ptr().cast::<u8>();
                let get_ptr =
                    |off: usize| std::ptr::read_unaligned(k.add(off).cast::<*mut c_void>());
                let get_u32 = |off: usize| std::ptr::read_unaligned(k.add(off).cast::<u32>());
                let handle = get_ptr(0);
                let grid = sys::dim3 { x: get_u32(8), y: get_u32(12), z: get_u32(16) };
                let block = sys::dim3 { x: get_u32(20), y: get_u32(24), z: get_u32(28) };
                let shared = get_u32(32);
                let kernel_params = get_ptr(40) as *mut *mut c_void;
                let function_type = get_u32(64);
                let mut cells = Vec::with_capacity(slots);
                for i in 0..slots {
                    // SAFETY: the kernelParams array is driver-owned memory
                    // associated with the node (V2 get docs); entries are
                    // the capture-time cell addresses — read as integers
                    // only, never dereferenced (capture-transient).
                    cells.push(read_slot_cell(kernel_params, i));
                }
                return Ok(NodeParamsReadback {
                    handle,
                    function_type,
                    grid,
                    block,
                    shared,
                    cells,
                    values: Vec::new(),
                    source: ReadbackSource::V2,
                    name: kernel_name(handle, function_type),
                });
            }
        }
    }
    // 2) Legacy driver read: the kernelParams array *and the argument
    //    values it points to* are owned by the node (cuda.h docs), so the
    //    values are directly readable. Only reached when the V2 symbol is
    //    unavailable (pre-13.2 runtimes) — on 13.x the legacy call returns
    //    success with a non-kernelParams encoding for library-launched
    //    nodes and must not be dereferenced (see the fn docs).
    //
    //    The encoding depends on the *driver* (it records the capture), not
    //    the runtime: a 13.x driver paired with a pre-13.2 runtime (V2
    //    symbol unavailable) takes this path with library-launched nodes
    //    recorded extra-style — the entries point at capture-transient
    //    memory and the value deref below segfaults (verified on this
    //    machine: driver 595.84 / CUDA 13.2 + 12.6 runtime, engine capture,
    //    first cublas node read crashed in `read_slot_value`). Fail closed
    //    on that combination instead of crashing; pre-13.0 drivers record
    //    classic kernelParams (the documented verified-working case).
    if let Some(major) = driver_version_major() {
        if major >= 13 {
            eprintln!(
                "reinfer-cuda graph: legacy node read-back unsafe — driver >= 13.0 records \
                 library-launched nodes extra-style and the V2 symbol is unavailable on this \
                 runtime; capture fails closed (eager)"
            );
            return Err(LaunchError::Fatal);
        }
    }
    let mut legacy: dsys::CUDA_KERNEL_NODE_PARAMS = unsafe { std::mem::zeroed() };
    // `_v2` is the only symbol in the 13.x driver ABI (cuda.h maps the
    // legacy name to it); cudarc gates the plain name to pre-12.x runtimes.
    let r = unsafe { dsys::cuGraphKernelNodeGetParams_v2(node as dsys::CUgraphNode, &mut legacy) };
    if r != dsys::CUresult::CUDA_SUCCESS {
        eprintln!(
            "reinfer-cuda graph: node read-back unsupported (V2 symbol unavailable and \
             legacy read failed with code={})",
            r as i32
        );
        return Err(LaunchError::Fatal);
    }
    let mut cells = Vec::with_capacity(slots);
    let mut values = Vec::with_capacity(slots);
    for i in 0..slots {
        // SAFETY: driver-owned array/value storage per the API docs
        // (read-only; never modified).
        unsafe {
            let cell = read_slot_cell(legacy.kernelParams, i);
            cells.push(cell);
            values.push(read_slot_value(legacy.kernelParams, i));
        }
    }
    Ok(NodeParamsReadback {
        handle: legacy.func as *mut c_void,
        function_type: FN_TYPE_FUNCTION,
        grid: sys::dim3 { x: legacy.gridDimX, y: legacy.gridDimY, z: legacy.gridDimZ },
        block: sys::dim3 { x: legacy.blockDimX, y: legacy.blockDimY, z: legacy.blockDimZ },
        shared: legacy.sharedMemBytes,
        cells,
        values,
        source: ReadbackSource::Legacy,
        name: kernel_name(legacy.func as *mut c_void, FN_TYPE_FUNCTION),
    })
}

/// Enumerate the kernel nodes of `graph` in creation order. Only handles and
/// node types are read: the legacy `cudaGraphKernelNodeGetParams` fails with
/// `cudaErrorInvalidDeviceFunction` for kernels loaded via
/// `cuLibraryLoadData` (new-style kernels expose parameters only through the
/// V2 generic node interface, which cudarc gates per CUDA version), so the
/// launch parameters come from the engine's [`KernelSpec`] declarations
/// instead — the count guard in `finish` verifies the shape matches.
fn enumerate_kernel_nodes(
    graph: sys::cudaGraph_t,
) -> Result<Vec<sys::cudaGraphNode_t>, LaunchError> {
    let mut count: usize = 0;
    // SAFETY: count slot valid; nodes may be null (size query).
    let r = unsafe { sys::cudaGraphGetNodes(graph, std::ptr::null_mut(), &mut count) };
    if r != sys::cudaError_t::cudaSuccess {
        return Err(graph_rc("cudaGraphGetNodes (size)", r));
    }
    let mut handles: Vec<sys::cudaGraphNode_t> = vec![std::ptr::null_mut(); count];
    // SAFETY: buffer sized `count`; count slot valid.
    let r = unsafe { sys::cudaGraphGetNodes(graph, handles.as_mut_ptr(), &mut count) };
    if r != sys::cudaError_t::cudaSuccess {
        return Err(graph_rc("cudaGraphGetNodes", r));
    }
    handles.truncate(count);

    let mut out = Vec::new();
    for h in handles {
        let mut ty = sys::cudaGraphNodeType::cudaGraphNodeTypeKernel;
        // SAFETY: type slot valid; node from this graph.
        let r = unsafe { sys::cudaGraphNodeGetType(h, &mut ty) };
        if r != sys::cudaError_t::cudaSuccess {
            return Err(graph_rc("cudaGraphNodeGetType", r));
        }
        if ty == sys::cudaGraphNodeType::cudaGraphNodeTypeKernel {
            out.push(h);
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// GraphExec
// ---------------------------------------------------------------------------

/// A captured, instantiated decode graph. Owns the exec + source graph
/// handles (the source graph is retained for `cudaGraphExecUpdate` and the
/// re-instantiate fallback) and the pool Arc (keeps the shared memory pool
/// and counters alive).
#[derive(Debug)]
pub struct GraphExec {
    pool: Arc<PoolInner>,
    exec: sys::cudaGraphExec_t,
    graph: sys::cudaGraph_t,
    nodes: Vec<KernelNode>,
    /// Per-kernel-node parameter read-back, in launch order (aligned with
    /// `nodes`); `None` for non-cublas (custom-kernel) nodes. See
    /// [`NodeParamsReadback`].
    readbacks: Vec<Option<NodeParamsReadback>>,
    /// Per-step content-refresh nodes (node indices with per-step scalar
    /// cells — gather token, rope q/k pos, kv_write phys/off): the driver
    /// bakes kernel params at capture (scalars frozen), so `replay`
    /// re-refreshes exactly these nodes every step (V2 SetParams) — the
    /// baked values are re-read from the cells at set time. All other
    /// nodes' params are constant (pointer targets are re-read through
    /// device memory) and stay baked.
    refresh: Vec<usize>,
    bucket: usize,
    seq_len: u32,
    workspace_bytes: u64,
}

impl GraphExec {
    /// Bucket index of this exec.
    #[must_use]
    pub fn bucket(&self) -> usize {
        self.bucket
    }

    /// Step length this exec was captured for (padding base).
    #[must_use]
    pub fn seq_len(&self) -> u32 {
        self.seq_len
    }

    /// Captured kernel node count (launch order).
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Workspace bytes this capture sub-allocated from the shared pool.
    #[must_use]
    pub fn workspace_bytes(&self) -> u64 {
        self.workspace_bytes
    }

    /// Driver read-back of the `index`-th captured kernel node's launch
    /// parameters (launch order; cublas nodes only — custom-kernel nodes
    /// return `None`). See [`NodeParamsReadback`] for what the read-back
    /// recovers on a given runtime: argument *values* on pre-13.2 runtimes
    /// (legacy read, driver-owned), metadata only on 13.x runtimes (V2
    /// generic read).
    #[must_use]
    pub fn node_params(&self, index: usize) -> Option<&NodeParamsReadback> {
        self.readbacks.get(index).and_then(|r| r.as_ref())
    }

    /// Seed the per-node staging arrays from the declaration's constant
    /// update list (Graph V2 — see the module docs "Replay without
    /// refresh"): the capture recorded the C3 cell addresses (the
    /// decl-driven launch passed `kernelParams[i] = &cell_i`), and the
    /// updates carry exactly those addresses, so after seeding the first
    /// replay is clean — no SetParams, no ExecUpdate, no re-instantiate;
    /// replay is a plain `cudaGraphLaunch` forever and the nodes read the
    /// cell *contents* (updated per step by `write_step`) at every launch.
    ///
    /// Must only be called when the capture really used the cell-args
    /// launch (an all-custom declaration); seeding over a stack-args
    /// capture would skip the refresh that would otherwise fail closed.
    pub fn seed_staging(&mut self, updates: &[PtrUpdate]) -> Result<(), LaunchError> {
        // Same full-coverage contract as `replay`.
        let coverage: Vec<Vec<usize>> = self.nodes.iter().map(|n| n.ptr_slots.clone()).collect();
        if !validate_updates(updates, &coverage) {
            eprintln!(
                "reinfer-cuda graph: seed_staging updates must cover every declared pointer \
                 slot of every node (nodes={}, ptr_slots={coverage:?})",
                self.nodes.len()
            );
            return Err(LaunchError::Fatal);
        }
        for u in updates {
            self.nodes[u.node].staging[u.slot] = u.ptr;
        }
        Ok(())
    }

    /// Replay the graph on `stream`. `updates` must cover every parameter
    /// slot of every node (see module docs). Pointer-diff refreshes are
    /// applied to the exec via `cudaGraphNodeSetParams` +
    /// `cudaGraphExecUpdate`; update failure falls back to destroy +
    /// re-instantiate and counts `reinstantiated`. Graph V2: the driver
    /// baked the kernel params at capture (scalars frozen — see module
    /// docs), so the per-step refresh nodes (gather token, rope q/k pos,
    /// kv_write phys/off) are re-refreshed from their cells every replay,
    /// unconditionally; the constant pointer updates then diff out
    /// (`seed_staging` + `write_step` keep them stable). Every call
    /// (successful or fallback) counts `graph_replay`.
    pub fn replay(
        &mut self,
        stream: &CudaStream,
        updates: &[PtrUpdate],
    ) -> Result<(), LaunchError> {
        // Coverage = the declared pointer slots of each node (not the whole
        // staging array — cublas nodes' geometry slots are never updated).
        let coverage: Vec<Vec<usize>> = self.nodes.iter().map(|n| n.ptr_slots.clone()).collect();
        if !validate_updates(updates, &coverage) {
            eprintln!(
                "reinfer-cuda graph: replay updates must cover every declared pointer slot \
                 of every node (nodes={}, ptr_slots={coverage:?})",
                self.nodes.len()
            );
            return Err(LaunchError::Fatal);
        }
        let state = self.pool.state.lock().unwrap_or_else(|p| p.into_inner());
        let counters = &state.buckets[self.bucket].counters;

        // Refresh-safety guard: a node whose read-back was V2-only (or
        // absent beyond the read cap) has staging slots that would launch
        // null — no refresh can make the exec launch-safe, so replay fails
        // closed up front, even when no pointer differs (a plain launch
        // would dereference the null cell addresses).
        if let Some((i, n)) = self.nodes.iter().enumerate().find(|(_, n)| !n.refresh_safe) {
            eprintln!(
                "reinfer-cuda graph: node {i} is not refresh-safe \
                 (read-back via {:?}, staging has unseeded slots); replay fails closed \
                 — the engine must fall back to eager",
                n.readback.as_ref().map(|r| r.source)
            );
            counters.eager_fallback.fetch_add(1, Ordering::Relaxed);
            return Err(LaunchError::Fatal);
        }

        // Apply the diffs into the staging arrays and detect changed nodes.
        let mut dirty = vec![false; self.nodes.len()];
        let mut any_dirty = false;
        for u in updates {
            let n = &mut self.nodes[u.node];
            let old = n.staging[u.slot];
            n.staging[u.slot] = u.ptr;
            if old != u.ptr {
                dirty[u.node] = true;
                any_dirty = true;
            }
        }
        // Graph V2 per-step content refresh: the driver baked the kernel
        // params at capture (scalars frozen), so the refresh nodes — the
        // ones whose cells `write_step` updates (gather token, rope q/k
        // pos, kv_write phys/off) — are re-refreshed from the cells every
        // replay, unconditionally (the SetParams re-bakes the current cell
        // contents). All other nodes' params are constant and stay baked.
        for &n in &self.refresh {
            dirty[n] = true;
            any_dirty = true;
        }
        if any_dirty {
            // Push the new pointer values into the source graph via the V2
            // generic `cudaGraphNodeSetParams` (only ptr slots are ever
            // touched; geometry/function stay captured), then sync the exec
            // with `cudaGraphExecUpdate` (pointer-diff only).
            //
            // The V2 generic interface is the only one the driver accepts
            // for kernels loaded via `cuLibraryLoadData`: the legacy
            // `cudaGraphKernelNodeSetParams`/`cudaGraphExecKernelNodeSetParams`
            // fail with `cudaErrorInvalidDeviceFunction`, and the handle the
            // node records is the `CUkernel` (`cudaKernelFunctionTypeKernel`),
            // which CUDA 13.x runtimes accept in the V2 union but 12.x
            // runtimes reject (they validate the handle as a `CUfunction` —
            // see `FN_TYPE`). The setter is the dlsym-resolved 13.x
            // implementation when the global runtime scope allows it
            // (`v2_node_set_params` — the LINKED symbol is versioned to the
            // build-time libcudart, so LD_PRELOAD cannot interpose it);
            // otherwise the linked symbol runs and the set fails permanently
            // (every attempt) — the replay fails closed (Err, no launch) and
            // `eager_fallback` counts the engine's fallback to eager.
            //
            // This dirty path is the fail-closed net: the Graph V2
            // production path re-refreshes the per-step nodes every replay
            // (see above) and the constant pointer updates diff out; the
            // path is exercised by the fake-step smoke tests and the
            // `REINFER_JGEMM=off` cublas boundary. When the 13.x V2
            // setter is unavailable (12.x global runtime scope) the set
            // fails permanently — replay fails closed and the engine
            // counts `eager_fallback`.
            let v2_set = v2_node_set_params();
            let mut set_failed: Option<sys::cudaError_t> = None;
            for (i, n) in self.nodes.iter_mut().enumerate() {
                if !dirty[i] {
                    continue;
                }
                let mut np = n.make_v2_params();
                // SAFETY: node belongs to our graph; params match the node's
                // function and staging length equals the declared arity; the
                // dlsym-resolved fn pointer matches the V2 ABI.
                let r = match v2_set {
                    Some(f) => unsafe { f(n.node, &mut np) },
                    None => unsafe { sys::cudaGraphNodeSetParams(n.node, &mut np) },
                };
                if r != sys::cudaError_t::cudaSuccess {
                    eprintln!(
                        "reinfer-cuda graph: cudaGraphNodeSetParams failed, code={}, \
                         replay fails closed (12.x global runtime scope rejects \
                         cuLibraryLoadData CUkernel handles; LD_PRELOAD libcudart.so.13 \
                         unlocks Graph V2 replay)",
                        r as i32
                    );
                    counters.eager_fallback.fetch_add(1, Ordering::Relaxed);
                    set_failed = Some(r);
                    break;
                }
            }
            if let Some(code) = set_failed {
                // The exec is untouched (still consistent with its last
                // successful pointers); do not launch with a half-refreshed
                // graph.
                return Err(graph_rc("cudaGraphNodeSetParams", code));
            }

            // Sync the exec with the graph via `cudaGraphExecUpdate`
            // (pointer-diff only — the graph already carries the fresh
            // pointers). Update failure (topology/function/type changed)
            // falls back to destroy + re-instantiate; the fresh exec
            // inherits the graph's current pointers.
            let mut info: sys::cudaGraphExecUpdateResultInfo_st = unsafe { std::mem::zeroed() };
            // SAFETY: `self.exec`/`self.graph` valid (created by us);
            // `info` output slot valid (the CUDA 12.4+ form updates the
            // exec in place and reports the result struct).
            let r = unsafe { sys::cudaGraphExecUpdate(self.exec, self.graph, &mut info) };
            if r == sys::cudaError_t::cudaSuccess
                && info.result
                    == sys::cudaGraphExecUpdateResult::cudaGraphExecUpdateSuccess
            {
                counters.exec_update_success.fetch_add(1, Ordering::Relaxed);
            } else {
                // ExecUpdate failed (topology/function/type changed, or
                // runtime error) — destroy + re-instantiate from the graph
                // (which already carries the fresh pointers). Both
                // `eager_fallback` and `reinstantiated` count the path
                // (the fallback is a degraded replay, not a plain launch).
                counters.eager_fallback.fetch_add(1, Ordering::Relaxed);
                // SAFETY: handle owned by this struct.
                let _ = unsafe { sys::cudaGraphExecDestroy(self.exec) }.result();
                self.exec = std::ptr::null_mut();
                // SAFETY: output slot valid; graph valid; flags=0.
                let mut fresh: sys::cudaGraphExec_t = std::ptr::null_mut();
                let r2 = unsafe { sys::cudaGraphInstantiate(&mut fresh, self.graph, 0) };
                if r2 != sys::cudaError_t::cudaSuccess {
                    return Err(graph_rc("cudaGraphInstantiate", r2));
                }
                self.exec = fresh;
                counters.reinstantiated.fetch_add(1, Ordering::Relaxed);
            }
        }

        // SAFETY: exec valid (possibly freshly re-instantiated); stream valid.
        let r = unsafe { sys::cudaGraphLaunch(self.exec, stream.handle()) };
        if r != sys::cudaError_t::cudaSuccess {
            return Err(graph_rc("cudaGraphLaunch", r));
        }
        counters.graph_replay.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

impl Drop for GraphExec {
    fn drop(&mut self) {
        if !self.exec.is_null() {
            // SAFETY: handle owned by this struct.
            let _ = unsafe { sys::cudaGraphExecDestroy(self.exec) }.result();
        }
        // SAFETY: handle owned by this struct.
        let _ = unsafe { sys::cudaGraphDestroy(self.graph) }.result();
    }
}

// ---------------------------------------------------------------------------
// Unit tests (pure logic — no GPU required)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bucket_index_boundaries() {
        // Below the minimum clamps to bucket 0.
        for s in [0, 1, 7] {
            assert_eq!(bucket_index(s), 0, "seq={s}");
        }
        // 8..=128 in steps of 8 (ceil mapping: 9..=16 -> bucket 1, cap 16).
        for (s, i) in [(8, 0usize), (9, 1), (15, 1), (16, 1), (17, 2), (127, 15), (128, 15)] {
            assert_eq!(bucket_index(s), i, "seq={s}");
        }
        // 129..=256 in steps of 16 (144 -> bucket 16, cap 144, padding 0).
        for (s, i) in [(129, 16usize), (143, 16), (144, 16), (145, 17), (255, 23), (256, 23)] {
            assert_eq!(bucket_index(s), i, "seq={s}");
        }
        // Above the maximum clamps to the last bucket.
        for s in [257, 1000, u32::MAX] {
            assert_eq!(bucket_index(s), BUCKET_COUNT - 1, "seq={s}");
        }
    }

    #[test]
    fn bucket_size_table() {
        for i in 0..BUCKET_SMALL {
            assert_eq!(bucket_size(i), 8 + 8 * i as u32, "small bucket {i}");
        }
        for i in BUCKET_SMALL..BUCKET_COUNT {
            assert_eq!(bucket_size(i), 128 + 16 * (i as u32 + 1 - BUCKET_SMALL as u32));
        }
        assert_eq!(bucket_size(0), 8);
        assert_eq!(bucket_size(15), 128);
        assert_eq!(bucket_size(16), 144);
        assert_eq!(bucket_size(BUCKET_COUNT - 1), 256);
    }

    #[test]
    fn bucket_capacity_covers_seq_len() {
        // Every captured step length (>= BUCKET_MIN) fits its bucket's
        // capacity; below BUCKET_MIN clamps to bucket 0.
        for s in BUCKET_MIN..=BUCKET_MAX {
            assert!(
                bucket_size(bucket_index(s)) >= s,
                "seq={s} bucket={} capacity={}",
                bucket_index(s),
                bucket_size(bucket_index(s))
            );
        }
        for s in [0, 1, BUCKET_MIN - 1] {
            assert_eq!(bucket_index(s), 0, "seq={s} clamps to bucket 0");
        }
    }

    #[test]
    fn padding_ratio_values() {
        assert_eq!(padding_ratio(8), 0.0);
        assert_eq!(padding_ratio(128), 0.0);
        assert_eq!(padding_ratio(256), 0.0);
        assert_eq!(padding_ratio(4), 0.5); // clamped to bucket 0: 4 of 8 idle
        assert_eq!(padding_ratio(0), 1.0); // clamped: all 8 idle
        assert_eq!(padding_ratio(9), 7.0 / 16.0);
        assert_eq!(padding_ratio(15), 1.0 / 16.0);
        assert_eq!(padding_ratio(129), 15.0 / 144.0);
        assert_eq!(padding_ratio(144), 0.0);
        // Clamped: never negative, never > 1.
        for s in 0..=300 {
            let r = padding_ratio(s);
            assert!((0.0..=1.0).contains(&r), "seq={s} ratio={r}");
        }
    }

    #[test]
    fn no_overlap_env_parsing() {
        assert!(no_overlap_from_env_value(None), "unset -> on");
        assert!(no_overlap_from_env_value(Some("")), "empty -> on");
        assert!(no_overlap_from_env_value(Some("1")));
        assert!(no_overlap_from_env_value(Some("true")));
        assert!(no_overlap_from_env_value(Some("on")));
        assert!(no_overlap_from_env_value(Some("garbage")));
        assert!(!no_overlap_from_env_value(Some("0")));
        assert!(!no_overlap_from_env_value(Some("false")));
        assert!(!no_overlap_from_env_value(Some("FALSE")));
        assert!(!no_overlap_from_env_value(Some("off")));
        assert!(!no_overlap_from_env_value(Some(" 0 ")));
    }

    #[test]
    fn pool_mb_env_parsing() {
        assert_eq!(pool_size_from_env_value(None), None);
        assert_eq!(pool_size_from_env_value(Some("")), None);
        assert_eq!(pool_size_from_env_value(Some("0")), None);
        assert_eq!(pool_size_from_env_value(Some("abc")), None);
        assert_eq!(pool_size_from_env_value(Some("64")), Some(64 * 1024 * 1024));
        assert_eq!(pool_size_from_env_value(Some("128")), Some(128 * 1024 * 1024));
        assert_eq!(pool_size_from_env_value(Some(" 16 ")), Some(16 * 1024 * 1024));
    }

    #[test]
    fn align_up_math() {
        assert_eq!(align_up(0, 256), 0);
        assert_eq!(align_up(1, 256), 256);
        assert_eq!(align_up(255, 256), 256);
        assert_eq!(align_up(256, 256), 256);
        assert_eq!(align_up(257, 256), 512);
        assert_eq!(align_up(12345, 256), 12544);
    }

    #[test]
    fn ptr_update_validation() {
        let u = |node: usize, slot: usize| PtrUpdate { node, slot, ptr: std::ptr::null_mut() };
        // Coverage = per-node sorted slot-index sets (cublas nodes only
        // declare their pointer slots — geometry slots are not in the set).
        let coverage: Vec<Vec<usize>> = vec![vec![0, 2], vec![0, 1, 3]];
        // Full coverage passes (slot sets need not be contiguous).
        assert!(validate_updates(&[u(0, 0), u(0, 2), u(1, 0), u(1, 1), u(1, 3)], &coverage));
        // Missing slot fails.
        assert!(!validate_updates(&[u(0, 0), u(1, 0), u(1, 1), u(1, 3)], &coverage));
        // Duplicate slot fails.
        assert!(!validate_updates(
            &[u(0, 0), u(0, 0), u(0, 2), u(1, 0), u(1, 1), u(1, 3)],
            &coverage
        ));
        // Undeclared slot (geometry slot, e.g. m) fails.
        assert!(!validate_updates(
            &[u(0, 0), u(0, 1), u(0, 2), u(1, 0), u(1, 1), u(1, 3)],
            &coverage
        ));
        // Out-of-range node fails.
        assert!(!validate_updates(&[u(2, 0), u(0, 2), u(1, 0), u(1, 1), u(1, 3)], &coverage));
        // Empty updates pass only for zero-coverage nodes.
        assert!(validate_updates(&[], &[]));
        assert!(validate_updates(&[], &[vec![]]));
        assert!(!validate_updates(&[], &coverage));
    }

    #[test]
    fn spec_validation() {
        let spec = |role: NodeRole, slots: usize, ptr_slots: Vec<(usize, PtrRole)>| KernelSpec {
            role,
            layout: ParamLayout::Fixed { slots },
            ptr_slots,
            handle: std::ptr::null_mut(),
            grid: sys::dim3 { x: 1, y: 1, z: 1 },
            block: sys::dim3 { x: 32, y: 1, z: 1 },
            shared: 0,
        };
        // Empty declaration fails closed.
        assert!(!validate_specs(&[]));
        // Custom kernel: every slot must be a pointer slot.
        let full = (0..4).map(|i| (i, PtrRole::Pointer)).collect::<Vec<_>>();
        assert!(validate_specs(&[spec(NodeRole::CustomKernel, 4, full.clone())]));
        assert!(!validate_specs(&[spec(NodeRole::CustomKernel, 4, full[..3].to_vec())]));
        assert!(!validate_specs(&[spec(NodeRole::CustomKernel, 4, vec![])]));
        // Memcpy role is reserved (rejected) even with full coverage.
        assert!(!validate_specs(&[spec(NodeRole::Memcpy, 4, full.clone())]));
        // Duplicate slot index fails.
        assert!(!validate_specs(&[spec(
            NodeRole::CustomKernel,
            4,
            vec![
                (0, PtrRole::Pointer),
                (0, PtrRole::Pointer),
                (2, PtrRole::Pointer),
                (3, PtrRole::Pointer)
            ]
        )]));
        // Out-of-range slot index fails.
        assert!(!validate_specs(&[spec(
            NodeRole::CustomKernel,
            4,
            vec![
                (4, PtrRole::Pointer),
                (1, PtrRole::Pointer),
                (2, PtrRole::Pointer),
                (3, PtrRole::Pointer)
            ]
        )]));
        // GEMM layout: partial pointer coverage is fine (geometry slots are
        // never refreshed).
        assert!(validate_specs(&[spec(
            NodeRole::CublasGemm,
            16,
            vec![(0, PtrRole::A), (1, PtrRole::B), (2, PtrRole::C)]
        )]));
        // Pointer slot outside the layout fails.
        assert!(!validate_specs(&[spec(NodeRole::CublasGemm, 16, vec![(16, PtrRole::A)])]));
    }

    #[test]
    fn spec_validation_gemm_mnk_readback() {
        // gemm_mnk recovers geometry only from a Legacy read-back, and only
        // at the slot indices declared by the layout.
        let layout = ParamLayout::Gemm { slots: 16, m: 0, n: 1, k: 2 };
        let rb = NodeParamsReadback {
            handle: std::ptr::null_mut(),
            function_type: 3,
            grid: sys::dim3 { x: 1, y: 1, z: 1 },
            block: sys::dim3 { x: 1, y: 1, z: 1 },
            shared: 0,
            cells: Vec::new(),
            values: vec![32, 64, 128],
            source: ReadbackSource::Legacy,
            name: None,
        };
        assert_eq!(rb.gemm_mnk(&layout), Some((32, 64, 128)));
        let v2 = NodeParamsReadback {
            values: vec![32, 64, 128],
            source: ReadbackSource::V2,
            ..rb.clone()
        };
        assert_eq!(v2.gemm_mnk(&layout), None, "V2 read-back has no values");
        assert_eq!(rb.gemm_mnk(&ParamLayout::Fixed { slots: 4 }), None);
        // Layout slots beyond the read values are None (not a panic).
        let short = NodeParamsReadback { values: vec![32], ..rb };
        assert_eq!(short.gemm_mnk(&layout), None);
    }

    #[test]
    fn disabled_pool_rejects_workspace_and_reports_zero() {
        // No GPU needed: the disabled slow path fails closed.
        let pool = GraphPool::disabled();
        assert!(!pool.enabled());
        assert_eq!(pool.memory_stats().pool_bytes, 0);
        assert_eq!(pool.summary().total.graph_replay, 0);
        assert_eq!(pool.summary().total.eager_fallback, 0);
        let err = pool.alloc_workspace(1024).expect_err("disabled pool must reject");
        assert_eq!(err, LaunchError::Fatal);
    }
}

// ---------------------------------------------------------------------------
// GPU smoke tests (real machine; #[ignore] + GPU-presence guard)
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "cuda"))]
mod ffi_tests {
    #![allow(clippy::unwrap_used)] // 测试断言崩溃即失败
    use super::*;
    use crate::CudaContext;
    use crate::buffer::{DeviceBuffer, HostBuffer, MemRef, copy};
    use crate::jit::{CtxGuard, JLib, KernelFn, launch_rows};
    use cudarc::cublas::sys as blas;
    use cudarc::driver::sys as dsys;
    use reinfer_jit::compile::{compile_cubin, gencode_flags};
    use reinfer_jit::{JitCache, JitKey, KernelSource, probe_toolchain_for_arch};

    /// Fake two-kernel decode step: `work = a + b` then `out = work * c`.
    /// 4 params per kernel (a, b, work, n) / (work, c, out, n) — pure
    /// elementwise math, deterministic (bitwise comparable).
    const FAKE_CU: &str = r#"
extern "C" __global__ void graph_fake_add(const float* a, const float* b, float* out, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { out[i] = a[i] + b[i]; }
}
extern "C" __global__ void graph_fake_scale(const float* a, const float* c, float* out, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { out[i] = a[i] * c[i]; }
}
"#;

    const N: u32 = 1 << 16;
    const BLOCK: u32 = 256;
    const GRID: u32 = N / BLOCK;
    const N_BYTES: usize = N as usize * 4;

    /// Stable argument cells: `PtrUpdate.ptr` values point at the *addresses
    /// of these fields* (the driver dereferences them at every launch). The
    /// test frame outlives all replays, so the addresses stay valid.
    #[derive(Clone, Copy)]
    struct Cells {
        a: *const f32,
        b: *const f32,
        c: *const f32,
        work: *mut f32,
        out: *mut f32,
        n: u32,
    }

    impl Cells {
        fn from_buffers(
            a: &DeviceBuffer,
            b: &DeviceBuffer,
            c: &DeviceBuffer,
            work: *mut f32,
            out: &DeviceBuffer,
        ) -> Self {
            Cells {
                a: a.as_ptr().cast(),
                b: b.as_ptr().cast(),
                c: c.as_ptr().cast(),
                work,
                out: out.as_ptr().cast_mut().cast(),
                n: N,
            }
        }
    }

    /// One fake decode step. Args packed as addresses of local variables
    /// (C3 discipline — the same rule applies inside the graph).
    struct FakeStep {
        add: KernelFn,
        scale: KernelFn,
        cells: Cells,
        dev: u32,
        /// Stable kernelParams arrays for the capture path (entry i = the
        /// address of the `i`-th arg cell). The driver records the array
        /// *pointer* at capture (measured on 595.84/CUDA 13.2: no copy —
        /// transient arrays are read as reused garbage at replay), so the
        /// arrays must live as long as the test — owned by the step.
        args0: [*mut c_void; 4],
        args1: [*mut c_void; 4],
    }

    impl FakeStep {
        fn new(add: KernelFn, scale: KernelFn, cells: Cells, dev: u32) -> Self {
            let args0 = [
                (&cells.a as *const *const f32) as *mut c_void,
                (&cells.b as *const *const f32) as *mut c_void,
                (&cells.work as *const *mut f32) as *mut c_void,
                (&cells.n as *const u32) as *mut c_void,
            ];
            let args1 = [
                (&cells.work as *const *mut f32) as *mut c_void,
                (&cells.c as *const *const f32) as *mut c_void,
                (&cells.out as *const *mut f32) as *mut c_void,
                (&cells.n as *const u32) as *mut c_void,
            ];
            FakeStep { add, scale, cells, dev, args0, args1 }
        }

        /// Launch with the stable args arrays built directly FROM the
        /// cells: `kernelParams[i] = &cell_i` (the stable C3 argument
        /// cells, not stack locals). The capture records the arrays'
        /// addresses as the node params — the durable-reference design:
        /// the arrays are owned by the step (never freed), their entries
        /// are the cell addresses, and clean launches read the cells
        /// forever — no SetParams is ever needed.
        fn run_via_cells(&self, stream: &CudaStream) -> Result<(), LaunchError> {
            // Copy the stable arrays to locals (the entries — the cell
            // addresses — are Copy; the arrays themselves stay owned by
            // the step so the capture-time recorded pointers stay valid).
            let mut args0 = self.args0;
            let mut args1 = self.args1;
            // SAFETY: test-owned kernels/buffers; context set by caller.
            unsafe {
                launch_rows(self.add, stream, self.dev, GRID, BLOCK, args0.as_mut_ptr())
            }?;
            // SAFETY: same as above.
            unsafe {
                launch_rows(self.scale, stream, self.dev, GRID, BLOCK, args1.as_mut_ptr())
            }?;
            Ok(())
        }

        fn run(&self, stream: &CudaStream) -> Result<(), LaunchError> {
            let a_v: *const f32 = self.cells.a;
            let b_v: *const f32 = self.cells.b;
            let work_v: *mut f32 = self.cells.work;
            let n_v: u32 = self.cells.n;
            let mut args0: [*mut c_void; 4] = [
                (&a_v as *const *const f32) as *mut c_void,
                (&b_v as *const *const f32) as *mut c_void,
                (&work_v as *const *mut f32) as *mut c_void,
                (&n_v as *const u32) as *mut c_void,
            ];
            // SAFETY: test-owned kernels/buffers; context set by caller.
            unsafe { launch_rows(self.add, stream, self.dev, GRID, BLOCK, args0.as_mut_ptr()) }?;

            let work2_v: *mut f32 = self.cells.work;
            let c_v: *const f32 = self.cells.c;
            let out_v: *mut f32 = self.cells.out;
            let n2_v: u32 = self.cells.n;
            let mut args1: [*mut c_void; 4] = [
                (&work2_v as *const *mut f32) as *mut c_void,
                (&c_v as *const *const f32) as *mut c_void,
                (&out_v as *const *mut f32) as *mut c_void,
                (&n2_v as *const u32) as *mut c_void,
            ];
            // SAFETY: same as above.
            unsafe { launch_rows(self.scale, stream, self.dev, GRID, BLOCK, args1.as_mut_ptr()) }?;
            Ok(())
        }
    }

    /// Per-slot `PtrUpdate`s referencing the stable cell addresses of `cells`
    /// (all 8 slots of the 2-node fake step).
    fn full_updates(cells: &Cells) -> Vec<PtrUpdate> {
        vec![
            PtrUpdate { node: 0, slot: 0, ptr: (&cells.a as *const *const f32) as *mut c_void },
            PtrUpdate { node: 0, slot: 1, ptr: (&cells.b as *const *const f32) as *mut c_void },
            PtrUpdate { node: 0, slot: 2, ptr: (&cells.work as *const *mut f32) as *mut c_void },
            PtrUpdate { node: 0, slot: 3, ptr: (&cells.n as *const u32) as *mut c_void },
            PtrUpdate { node: 1, slot: 0, ptr: (&cells.work as *const *mut f32) as *mut c_void },
            PtrUpdate { node: 1, slot: 1, ptr: (&cells.c as *const *const f32) as *mut c_void },
            PtrUpdate { node: 1, slot: 2, ptr: (&cells.out as *const *mut f32) as *mut c_void },
            PtrUpdate { node: 1, slot: 3, ptr: (&cells.n as *const u32) as *mut c_void },
        ]
    }

    /// Eager reference run: launch the step on the stream, sync, copy out.
    fn run_eager(
        step: &FakeStep,
        stream: &CudaStream,
        dev: u32,
        out: &DeviceBuffer,
        hout: &HostBuffer,
    ) -> Vec<f32> {
        let _guard = CtxGuard::set_current(dev).expect("guard");
        step.run(stream).expect("eager step");
        stream.synchronize().expect("sync");
        copy(&mut MemRef::Host(hout), &mut MemRef::Device(out), N_BYTES, None).expect("d2h");
        snapshot(hout, N as usize)
    }

    fn snapshot(buf: &HostBuffer, n: usize) -> Vec<f32> {
        // SAFETY: read-only, sized `n` f32s.
        unsafe { core::slice::from_raw_parts(buf.as_ptr() as *const f32, n).to_vec() }
    }

    fn fill(buf: &HostBuffer, data: &[f32]) {
        // SAFETY: pinned host buffer sized for `data` elements.
        unsafe {
            let s = core::slice::from_raw_parts_mut(buf.as_ptr() as *mut f32, data.len());
            s.copy_from_slice(data);
        }
    }

    fn to_dev(dev: DeviceId, h: &HostBuffer) -> DeviceBuffer {
        let db = DeviceBuffer::alloc(dev, h.size()).expect("dev buf");
        copy(&mut MemRef::Device(&db), &mut MemRef::Host(h), h.size(), None).expect("h2d");
        db
    }

    fn has_gpu() -> bool {
        CudaContext::device_count().unwrap_or(0) >= 1
    }

    fn setup() -> (CudaContext, CudaStream) {
        let ctx = CudaContext::init(DeviceId::new(0)).expect("ctx");
        let stream = CudaStream::new(ctx.device_id()).expect("stream");
        (ctx, stream)
    }

    fn fake_kernels(cache_dir: &std::path::Path) -> (JLib, KernelFn, KernelFn) {
        let arch = crate::arch::resolve_arch().expect("arch");
        let tc = probe_toolchain_for_arch(&arch).expect("toolchain");
        let src = KernelSource {
            name: "graph_fake",
            src: FAKE_CU,
            headers: vec![],
            flags: gencode_flags(&arch).expect("gencode"),
            arch: arch.clone(),
            toolchain_ver: tc.ver_line.clone(),
        };
        let cache = JitCache::open(Some(cache_dir.to_path_buf())).expect("cache");
        let key = JitKey::new(&src, &tc);
        let (_, cubin_path) =
            cache.build_once(&key, &src, || compile_cubin(&src, &tc)).expect("build");
        let bytes = std::fs::read(&cubin_path).expect("cubin");
        let lib = JLib::from_bytes(bytes).expect("lib");
        let add = lib.kernel("graph_fake_add").expect("kernel add");
        let scale = lib.kernel("graph_fake_scale").expect("kernel scale");
        (lib, add, scale)
    }

    fn make_hosts() -> (HostBuffer, HostBuffer, HostBuffer, HostBuffer) {
        (
            HostBuffer::alloc(N_BYTES).expect("ha"),
            HostBuffer::alloc(N_BYTES).expect("hb"),
            HostBuffer::alloc(N_BYTES).expect("hc"),
            HostBuffer::alloc(N_BYTES).expect("hout"),
        )
    }

    /// Bitwise equality: f32 bits must match exactly.
    fn assert_bitwise(got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert_eq!(g.to_bits(), w.to_bits(), "slot {i}: {g} vs {w}");
        }
    }

    /// Replay output into `out` and compare bitwise against `want`.
    fn assert_replay_output(
        exec: &mut GraphExec,
        stream: &CudaStream,
        updates: &[PtrUpdate],
        out: &DeviceBuffer,
        hout: &HostBuffer,
        want: &[f32],
    ) {
        exec.replay(stream, updates).expect("replay");
        stream.synchronize().expect("sync");
        copy(&mut MemRef::Host(hout), &mut MemRef::Device(out), N_BYTES, None).expect("d2h");
        assert_bitwise(&snapshot(hout, N as usize), want);
    }

    /// Engine-declared specs for the two fake kernels (4 fixed 8-byte slots
    /// each — all pointer slots; geometry matches `FakeStep::run`). The
    /// handle is the `CUkernel` the driver records for `cuLibraryLoadData`
    /// kernels (`cudaKernelFunctionTypeKernel`), fetched via
    /// `cuLibraryGetKernel`.
    fn specs_for(lib: &JLib) -> [KernelSpec; 2] {
        let kern = |name: &str| -> *mut c_void {
            let cname = std::ffi::CString::new(name).expect("cstring");
            let mut k: dsys::CUkernel = std::ptr::null_mut();
            // SAFETY: test-owned library; output slot valid; NUL-terminated.
            let r = unsafe { dsys::cuLibraryGetKernel(&mut k, lib.raw(), cname.as_ptr()) };
            assert_eq!(r, dsys::CUresult::CUDA_SUCCESS, "cuLibraryGetKernel {name}");
            k as *mut c_void
        };
        let grid = sys::dim3 { x: GRID, y: 1, z: 1 };
        let block = sys::dim3 { x: BLOCK, y: 1, z: 1 };
        let layout = ParamLayout::Fixed { slots: 4 };
        let ptr_slots = (0..4).map(|i| (i, PtrRole::Pointer)).collect::<Vec<_>>();
        let make = |handle: *mut c_void| KernelSpec {
            role: NodeRole::CustomKernel,
            layout: layout.clone(),
            ptr_slots: ptr_slots.clone(),
            handle,
            grid,
            block,
            shared: 0,
        };
        [make(kern("graph_fake_add")), make(kern("graph_fake_scale"))]
    }

    /// Capture the fake step on `pool` for `seq_len` (workspace from the
    /// shared pool) and return the exec. Specs go through the
    /// [`GraphPool::declare_specs`] surface (the engine path — declared
    /// before the capture window opens, consumed by the next `capture`);
    /// the capture itself passes an empty slice.
    fn capture_fake_step(
        pool: &GraphPool,
        stream: &CudaStream,
        lib: &JLib,
        add: KernelFn,
        scale: KernelFn,
        dev: u32,
        seq_len: u32,
        cells: &Cells,
    ) -> GraphExec {
        pool.declare_specs(specs_for(lib).to_vec()).expect("declare_specs");
        let step = FakeStep::new(add, scale, *cells, dev);
        let closure = |s: &CudaStream| {
            let _g = CtxGuard::set_current(dev).expect("guard");
            step.run(s)
        };
        pool.capture(stream, seq_len, &[], &[], closure).expect("capture")
    }

    /// Graph capture + replay vs eager: bitwise-identical outputs for
    /// same-pointer replay and for new-pointer (ExecUpdate refresh) replay,
    /// with shared-pool memory accounting and counter assertions.
    ///
    /// Runtime-adaptive: pointer-only refresh of `cuLibraryLoadData` kernel
    /// nodes needs the CUDA >= 13 runtime (`cudaGraphNodeSetParams` V2 with
    /// `CUkernel` handles — see module docs). On 12.x runtimes the refresh
    /// fails permanently, so the test asserts the fail-closed path instead
    /// (replay Err, no launch, `eager_fallback` counted).
    #[test]
    #[ignore = "gpu.yml: graph-smoke"]
    fn graph_replay_matches_eager_bitwise() {
        if !has_gpu() {
            eprintln!("skip: no GPU");
            return;
        }
        let v2 = v2_params_available();
        let (ctx, stream) = setup();
        let dev = ctx.device_id();
        let cache = std::env::temp_dir().join("reinfer-jit-graph-smoke");
        let _ = std::fs::remove_dir_all(&cache);
        let (lib, add, scale) = fake_kernels(&cache);

        let (ha, hb, hc, hout) = make_hosts();
        fill(&ha, &(0..N as usize).map(|i| (i as f32 * 0.001).sin()).collect::<Vec<_>>());
        fill(&hb, &(0..N as usize).map(|i| (i as f32 * 0.0005).cos()).collect::<Vec<_>>());
        fill(&hc, &vec![1.25; N as usize]);
        let a = to_dev(dev, &ha);
        let b = to_dev(dev, &hb);
        let c = to_dev(dev, &hc);
        let out = DeviceBuffer::alloc(dev, N_BYTES).expect("out");

        // Graph path: work buffer sub-allocated from the shared pool.
        let pool = GraphPool::new(dev);
        let ws = pool.alloc_workspace(N_BYTES).expect("pool workspace");
        let cells1 = Cells::from_buffers(&a, &b, &c, ws.ptr().cast(), &out);
        let mut exec = capture_fake_step(&pool, &stream, &lib, add, scale, dev.index(), 8, &cells1);
        assert_eq!(exec.bucket(), 0);
        assert_eq!(exec.node_count(), 2);

        if v2 {
            eprintln!("V2 node-params available: asserting the full pointer-refresh path");
            // Eager reference: same kernel sequence, same data, plain buffers.
            let work_eager = DeviceBuffer::alloc(dev, N_BYTES).expect("work eager");
            let cells_eager =
                Cells::from_buffers(&a, &b, &c, work_eager.as_ptr().cast_mut().cast(), &out);
            let step_eager = FakeStep::new(add, scale, cells_eager, dev.index());
            let want = run_eager(&step_eager, &stream, dev.index(), &out, &hout);

            // Replay 1: cell addresses (fresh after capture; staging was null).
            assert_replay_output(&mut exec, &stream, &full_updates(&cells1), &out, &hout, &want);

            // Replay 2: brand-new buffers with different values -> pointer-diff
            // refresh (different cell addresses) must be reflected in the output.
            let a2 = to_dev(dev, &ha);
            let b2 = to_dev(dev, &hb);
            let c2 = to_dev(dev, &hc);
            let out2 = DeviceBuffer::alloc(dev, N_BYTES).expect("out2");
            let cells_eager2 =
                Cells::from_buffers(&a2, &b2, &c2, work_eager.as_ptr().cast_mut().cast(), &out2);
            let step_eager2 = FakeStep::new(add, scale, cells_eager2, dev.index());
            let want2 = run_eager(&step_eager2, &stream, dev.index(), &out2, &hout);
            let cells2 = Cells::from_buffers(&a2, &b2, &c2, ws.ptr().cast(), &out2);
            assert_replay_output(&mut exec, &stream, &full_updates(&cells2), &out2, &hout, &want2);

            // Counters: bucket 0 only; both replays took the pointer-diff path.
            let c0 = pool.bucket_counters(0);
            assert_eq!(c0.graph_replay, 2);
            assert_eq!(c0.eager_fallback, 0);
            assert_eq!(c0.reinstantiated, 0);
            assert_eq!(c0.exec_update_success, 2);
            assert_eq!(c0.padding_ratio, 0.0, "seq_len=8 -> bucket 0 (capacity 8)");

            // Summary aggregate matches per-bucket.
            let sum = pool.summary();
            assert_eq!(sum.total.graph_replay, c0.graph_replay);
            assert_eq!(sum.total.eager_fallback, 0);
        } else {
            // 12.x runtime: cudaGraphNodeSetParams rejects cuLibraryLoadData
            // kernel nodes permanently (verified by exhaustive sweep; see
            // module docs) -> the pointer-diff refresh fails, replay must Err
            // and the engine falls back to eager.
            eprintln!(
                "V2 node-params unavailable: pointer-diff refresh of cuLibraryLoadData \
                 kernel nodes is unsupported -> asserting the fail-closed path"
            );
            let _err = exec
                .replay(&stream, &full_updates(&cells1))
                .expect_err("12.x runtime: replay must fail closed (no launch)");
            let c0 = pool.bucket_counters(0);
            assert_eq!(c0.graph_replay, 0, "no launch happened on fail-closed replay");
            assert_eq!(c0.eager_fallback, 1, "fail-closed replay counted as eager fallback");
            assert_eq!(c0.reinstantiated, 0, "set failure keeps the exec consistent");
            assert_eq!(c0.exec_update_success, 0);
            let sum = pool.summary();
            assert_eq!(sum.total.graph_replay, 0);
            assert_eq!(sum.total.eager_fallback, 1);
        }

        // Memory accounting: measured pool, print profile, <= 8% gate.
        let mem = pool.memory_stats();
        println!(
            "<GraphPool>= {} bytes (requested), measured delta = {} bytes, \
             total device = {} bytes, fraction = {:.5}, workspace used = {} bytes",
            mem.pool_bytes,
            mem.measured_delta_bytes,
            mem.total_device_bytes,
            mem.pool_fraction,
            mem.workspace_used_bytes
        );
        assert!(mem.pool_bytes > 0, "pool allocated");
        assert!(mem.measured_delta_bytes > 0, "measured allocation delta");
        assert_eq!(mem.workspace_used_bytes, N_BYTES, "pool sub-allocation accounting");
        assert!(
            mem.pool_fraction <= 0.08,
            "pool fraction {:.4} must stay <= 8%",
            mem.pool_fraction
        );
    }

    /// Durable-params probe: capture the step with the args arrays built
    /// FROM the cells (`kernelParams[i] = &cell_i`), seed the staging from
    /// the same cell addresses so the replays are never dirty (no SetParams,
    /// no ExecUpdate, no re-instantiation), then run clean replays with the
    /// buffer contents mutated in place (the engine's per-step cell writes).
    /// If the clean launches stay correct, the driver's capture-time param
    /// storage is durable and the engine's graph needs NO per-step refresh —
    /// the C3 cells themselves are the graph's argument source.
    #[test]
    #[ignore = "gpu.yml: graph-smoke"]
    fn capture_baked_cell_args_stay_correct() {
        if !has_gpu() {
            eprintln!("skip: no GPU");
            return;
        }
        if !v2_params_available() {
            eprintln!("skip: no V2 node-params (fail-closed path asserted elsewhere)");
            return;
        }
        let (ctx, stream) = setup();
        let dev = ctx.device_id();
        let cache = std::env::temp_dir().join("reinfer-jit-graph-cells");
        let _ = std::fs::remove_dir_all(&cache);
        let (lib, add, scale) = fake_kernels(&cache);

        let (ha, hb, hc, hout) = make_hosts();
        let (ha2, hb2, hc2, hout2) = make_hosts();
        fill(&ha, &vec![1.0; N as usize]);
        fill(&hb, &vec![2.0; N as usize]);
        fill(&hc, &vec![3.0; N as usize]);
        fill(&ha2, &vec![1.5; N as usize]);
        fill(&hb2, &vec![2.5; N as usize]);
        fill(&hc2, &vec![3.5; N as usize]);
        let a = to_dev(dev, &ha);
        let b = to_dev(dev, &hb);
        let c = to_dev(dev, &hc);
        let out = DeviceBuffer::alloc(dev, N_BYTES).expect("out");
        let work = DeviceBuffer::alloc(dev, N_BYTES).expect("work");

        let pool = GraphPool::new(dev);
        let cells = Cells::from_buffers(&a, &b, &c, work.as_ptr().cast_mut().cast(), &out);
        // Capture with the cell-address args (run_via_cells).
        pool.declare_specs(specs_for(&lib).to_vec()).expect("declare_specs");
        let step = FakeStep::new(add, scale, cells, dev.index());
        let closure = |s: &CudaStream| {
            let _g = CtxGuard::set_current(dev.index()).expect("guard");
            step.run_via_cells(s)
        };
        let mut exec = pool.capture(&stream, 8, &[], &[], closure).expect("capture");

        // Seed the staging from the same cell addresses (what the capture
        // recorded — the `GraphExec::seed_staging` surface): the updates
        // then never dirty — pure clean launches.
        exec.seed_staging(&full_updates(&cells)).expect("seed staging");

        let eager = |hout: &HostBuffer| {
            let cells_eager =
                Cells::from_buffers(&a, &b, &c, work.as_ptr().cast_mut().cast(), &out);
            let step = FakeStep::new(add, scale, cells_eager, dev.index());
            run_eager(&step, &stream, dev.index(), &out, hout)
        };

        let want1 = eager(&hout);
        assert_replay_output(&mut exec, &stream, &full_updates(&cells), &out, &hout, &want1);

        // Engine steady state: same addresses, new contents.
        copy(&mut MemRef::Device(&a), &mut MemRef::Host(&ha2), N_BYTES, None).expect("h2d a2");
        copy(&mut MemRef::Device(&b), &mut MemRef::Host(&hb2), N_BYTES, None).expect("h2d b2");
        copy(&mut MemRef::Device(&c), &mut MemRef::Host(&hc2), N_BYTES, None).expect("h2d c2");
        stream.synchronize().expect("sync after h2d");
        let want2 = eager(&hout2);
        assert_replay_output(&mut exec, &stream, &full_updates(&cells), &out, &hout2, &want2);
        let want2b = eager(&hout);
        assert_replay_output(&mut exec, &stream, &full_updates(&cells), &out, &hout, &want2b);

        let c0 = pool.bucket_counters(0);
        println!(
            "cell-args-capture: replays {}, exec_update_success {}, eager_fallback {}, \
             reinstantiated {}",
            c0.graph_replay, c0.exec_update_success, c0.eager_fallback, c0.reinstantiated
        );
        assert_eq!(c0.exec_update_success, 0, "clean replays must never refresh");
        assert_eq!(c0.eager_fallback, 0);
        assert_eq!(c0.reinstantiated, 0);
    }

    /// ExecUpdate failure path: poison the source graph with an extra empty
    /// node (topology change), replay with dirty cells -> cudaGraphExecUpdate
    /// fails (TopologyChanged) -> destroy + re-instantiate fallback (the
    /// graph already carries the fresh pointers, so the re-instantiated exec
    /// produces correct output) and counters count the fallback.
    ///
    /// Runtime-adaptive: on 12.x runtimes the first refresh already fails
    /// closed (see module docs), so the test asserts that path instead and
    /// skips the poison step.
    #[test]
    #[ignore = "gpu.yml: graph-smoke"]
    fn exec_update_failure_reinstantiates() {
        if !has_gpu() {
            eprintln!("skip: no GPU");
            return;
        }
        let v2 = v2_params_available();
        let (ctx, stream) = setup();
        let dev = ctx.device_id();
        let cache = std::env::temp_dir().join("reinfer-jit-graph-fail");
        let _ = std::fs::remove_dir_all(&cache);
        let (lib, add, scale) = fake_kernels(&cache);

        let (ha, hb, hc, hout) = make_hosts();
        fill(&ha, &vec![1.0; N as usize]);
        fill(&hb, &vec![2.0; N as usize]);
        fill(&hc, &vec![3.0; N as usize]);
        let a = to_dev(dev, &ha);
        let b = to_dev(dev, &hb);
        let c = to_dev(dev, &hc);
        let out = DeviceBuffer::alloc(dev, N_BYTES).expect("out");
        let work = DeviceBuffer::alloc(dev, N_BYTES).expect("work");

        let pool = GraphPool::new(dev);
        let cells1 = Cells::from_buffers(&a, &b, &c, work.as_ptr().cast_mut().cast(), &out);
        let mut exec = capture_fake_step(&pool, &stream, &lib, add, scale, dev.index(), 8, &cells1);

        if v2 {
            eprintln!("V2 node-params available: asserting the full ExecUpdate/re-instantiate path");
            // Reference: eager (work = a+b = 3; out = work*c = 9 everywhere).
            let cells_eager =
                Cells::from_buffers(&a, &b, &c, work.as_ptr().cast_mut().cast(), &out);
            let step_eager = FakeStep::new(add, scale, cells_eager, dev.index());
            let want = run_eager(&step_eager, &stream, dev.index(), &out, &hout);

            // Clean replay: pointer-diff refresh succeeds.
            assert_replay_output(&mut exec, &stream, &full_updates(&cells1), &out, &hout, &want);
            assert_eq!(pool.bucket_counters(0).exec_update_success, 1);
            assert_eq!(pool.bucket_counters(0).eager_fallback, 0);

            // Poison: add an empty node to the source graph (pure topology
            // change — no kernel involved, so it cannot hit the JIT-kernel
            // legacy-params limitation). ExecUpdate must report TopologyChanged.
            let mut extra_node: sys::cudaGraphNode_t = std::ptr::null_mut();
            // SAFETY: test-owned graph handle; no dependencies.
            let r = unsafe {
                sys::cudaGraphAddEmptyNode(&mut extra_node, exec.graph, std::ptr::null_mut(), 0)
            };
            assert_eq!(r, sys::cudaError_t::cudaSuccess, "poison node add");

            // Replay 2 with *different* buffers and values (dirty slots):
            // SetParams on the graph succeeds with the fresh pointers, then
            // ExecUpdate must fail (TopologyChanged) -> destroy + re-instantiate
            // fallback; the re-instantiated exec inherits the graph's fresh
            // pointers, so the output is still correct.
            let (ha2, hb2, hc2, hout2) = make_hosts();
            fill(&ha2, &vec![1.5; N as usize]);
            fill(&hb2, &vec![2.5; N as usize]);
            fill(&hc2, &vec![3.5; N as usize]);
            let a2 = to_dev(dev, &ha2);
            let b2 = to_dev(dev, &hb2);
            let c2 = to_dev(dev, &hc2);
            let out2 = DeviceBuffer::alloc(dev, N_BYTES).expect("out2");
            let work2 = DeviceBuffer::alloc(dev, N_BYTES).expect("work2");
            let cells_eager2 =
                Cells::from_buffers(&a2, &b2, &c2, work2.as_ptr().cast_mut().cast(), &out2);
            let step_eager2 = FakeStep::new(add, scale, cells_eager2, dev.index());
            let want2 = run_eager(&step_eager2, &stream, dev.index(), &out2, &hout2);
            let cells2 =
                Cells::from_buffers(&a2, &b2, &c2, work2.as_ptr().cast_mut().cast(), &out2);
            assert_replay_output(&mut exec, &stream, &full_updates(&cells2), &out2, &hout2, &want2);

            let c0 = pool.bucket_counters(0);
            assert_eq!(c0.graph_replay, 2, "both replays counted");
            assert_eq!(c0.eager_fallback, 1, "ExecUpdate failure counted");
            assert_eq!(c0.reinstantiated, 1, "re-instantiate fallback ran");
            assert_eq!(c0.exec_update_success, 1, "only the clean replay refreshed via ExecUpdate");
        } else {
            // 12.x runtime: the refresh already fails closed on the very first
            // dirty replay (cudaGraphNodeSetParams rejects cuLibraryLoadData
            // kernel nodes) — assert that path; the poison/re-instantiate
            // sequence is unreachable here (nothing ever got updated).
            eprintln!(
                "V2 node-params unavailable: pointer-diff refresh unsupported -> \
                 asserting the fail-closed path (poison step skipped)"
            );
            let _err = exec
                .replay(&stream, &full_updates(&cells1))
                .expect_err("12.x runtime: replay must fail closed (no launch)");
            let c0 = pool.bucket_counters(0);
            assert_eq!(c0.graph_replay, 0, "no launch happened");
            assert_eq!(c0.eager_fallback, 1, "fail-closed replay counted as eager fallback");
            assert_eq!(c0.reinstantiated, 0, "set failure does not re-instantiate");
            assert_eq!(c0.exec_update_success, 0);
        }
    }

    /// Per-bucket `padding_ratio` recording at capture time.
    #[test]
    #[ignore = "gpu.yml: graph-smoke"]
    fn padding_ratio_recorded_per_bucket() {
        if !has_gpu() {
            eprintln!("skip: no GPU");
            return;
        }
        let (ctx, stream) = setup();
        let dev = ctx.device_id();
        let cache = std::env::temp_dir().join("reinfer-jit-graph-pad");
        let _ = std::fs::remove_dir_all(&cache);
        let (lib, add, scale) = fake_kernels(&cache);

        let (ha, hb, _hc, _hout) = make_hosts();
        fill(&ha, &vec![0.5; N as usize]);
        fill(&hb, &vec![0.25; N as usize]);
        let a = to_dev(dev, &ha);
        let b = to_dev(dev, &hb);
        let work = DeviceBuffer::alloc(dev, N_BYTES).expect("work");
        let out = DeviceBuffer::alloc(dev, N_BYTES).expect("out");
        let c = DeviceBuffer::alloc(dev, N_BYTES).expect("c");

        let pool = GraphPool::new(dev);
        let cells = Cells::from_buffers(&a, &b, &c, work.as_ptr().cast_mut().cast(), &out);

        // seq_len=10 -> bucket 1 (capacity 16): padding (16-10)/16 = 0.375.
        let step = |cells: &Cells| {
            let step = FakeStep::new(add, scale, *cells, dev.index());
            move |s: &CudaStream| {
                let _g = CtxGuard::set_current(dev.index()).expect("guard");
                step.run(s)
            }
        };
        pool.declare_specs(specs_for(&lib).to_vec()).expect("declare_specs");
        let exec = pool.capture(&stream, 10, &[], &[], &step(&cells)).expect("capture");
        assert_eq!(exec.bucket(), 1);
        assert_eq!(pool.bucket_counters(1).padding_ratio, 0.375);
        // Capturing again with a different length overwrites the ratio.
        pool.declare_specs(specs_for(&lib).to_vec()).expect("declare_specs 2");
        let _exec2 = pool.capture(&stream, 9, &[], &[], &step(&cells)).expect("capture 2");
        assert_eq!(pool.bucket_counters(1).padding_ratio, (16.0 - 9.0) / 16.0);
    }

    /// BLOCKER-A step-b on a real machine (bench/notes.md): capture two
    /// engine-identical cublas GEMM launches, count the graph's kernel
    /// nodes, and read each node's launch parameters back from the driver.
    ///
    /// Acceptance: capture completes after the spec declaration (the "0
    /// declared specs" fail-closed guard no longer fires), the node-count
    /// guard matches, and the read-back conclusion is recorded: on
    /// pre-13.2 runtimes (e.g. the machine's default 12.6) the legacy
    /// `cuGraphKernelNodeGetParams` serves driver-owned argument *values*
    /// (m/n/k geometry, alpha, operand staging pointers); on 13.x runtimes
    /// the V2 generic `cudaGraphNodeGetParams` serves the kernel member
    /// only (metadata; values are capture-time cell addresses). When the
    /// legacy source was available, a same-pointer replay must be bitwise
    /// equal to eager; on a V2-only read-back the replay must fail closed
    /// (unseeded staging).
    #[test]
    #[ignore = "gpu.yml: graph-cublas-readback"]
    fn cublas_gemm_node_readback() {
        if !has_gpu() {
            eprintln!("skip: no GPU");
            return;
        }
        let (major, minor) = runtime_version().expect("runtime version");
        let (ctx, stream) = setup();
        let dev = ctx.device_id();
        let pool = GraphPool::new(dev);
        let gemm = crate::gemm::Gemm::new(dev.index()).expect("cublas handle");

        // Two shapes, f16 operands / f32 output (the engine's gate compute
        // config): A [m x k] row-major, B [k x n] row-major, C [m x n].
        let shapes: [(i32, i32, i32); 2] = [(32, 64, 128), (16, 48, 96)];
        let mut bufs: Vec<(DeviceBuffer, DeviceBuffer, DeviceBuffer)> = Vec::new();
        for (m, n, k) in shapes {
            let a = DeviceBuffer::alloc(dev, m as usize * k as usize * 2).expect("A");
            let b = DeviceBuffer::alloc(dev, n as usize * k as usize * 2).expect("B");
            let c = DeviceBuffer::alloc(dev, m as usize * n as usize * 4).expect("C");
            bufs.push((a, b, c));
        }
        let a16 = blas::cudaDataType_t::CUDA_R_16F;
        let f32 = blas::cudaDataType_t::CUDA_R_32F;
        let gmat = |ptr: *mut c_void, dtype: blas::cudaDataType_t, ld: i32| crate::gemm::GpuMat {
            ptr,
            dtype,
            ld,
        };
        let run_gemms = |stream: &CudaStream| -> Result<(), LaunchError> {
            let _g = CtxGuard::set_current(dev.index()).expect("guard");
            for (i, (m, n, k)) in shapes.into_iter().enumerate() {
                let (a, b, c) = &bufs[i];
                let mut gm = gmat(c.as_ptr().cast_mut().cast(), f32, m);
                gemm.gemm_f32acc(
                    stream,
                    m,
                    n,
                    k,
                    &gmat(a.as_ptr().cast_mut().cast(), a16, k),
                    &gmat(b.as_ptr().cast_mut().cast(), a16, k),
                    &mut gm,
                    1.0,
                    0.0,
                )
                .expect("gemm");
            }
            Ok(())
        };

        // Engine-declared specs: cublas GEMM role; 16 slots read (the
        // read-back cap); geometry at slots 0/1/2 (m/n/k); NO declared
        // pointer slots — the cublas-internal argument layout is unknown to
        // the engine, so the read-back is the discovery surface (the slot
        // roles A/B/C land wherever the cublas kernel packs them, or behind
        // cublas staging). Handle/grid/block are unknown at declaration
        // time (cublas-internal kernel) — declared null/zero and recovered
        // by the read-back.
        let specs: Vec<KernelSpec> = shapes
            .into_iter()
            .map(|_| KernelSpec {
                role: NodeRole::CublasGemm,
                layout: ParamLayout::Gemm { slots: 16, m: 0, n: 1, k: 2 },
                ptr_slots: vec![],
                handle: std::ptr::null_mut(),
                grid: sys::dim3 { x: 0, y: 0, z: 0 },
                block: sys::dim3 { x: 0, y: 0, z: 0 },
                shared: 0,
            })
            .collect();
        pool.declare_specs(specs).expect("declare_specs");

        // Capture window: two cublas GEMM launches (engine-identical call
        // sequence). Must succeed — the spec declaration lifts the
        // "0 declared specs" fail-closed error.
        let mut exec = pool
            .capture(&stream, 8, &[], &[], &run_gemms)
            .expect("capture with declared specs must succeed");
        assert_eq!(exec.node_count(), 2, "two cublas kernel nodes");

        // The fail-closed guard is still active: a capture with no
        // declaration (empty slice, nothing pending) fails before the
        // window opens.
        let pool2 = GraphPool::new(dev);
        let err = pool2
            .capture(&stream, 8, &[], &[], |_| Ok(()))
            .expect_err("0 declared specs must fail closed");
        assert_eq!(err, LaunchError::Fatal);

        // Read-back per node: source, handle, geometry, recoverability.
        let mut legacy_count = 0usize;
        let mut values_ok = true;
        for (i, (m, n, k)) in shapes.into_iter().enumerate() {
            let rb = exec.node_params(i).expect("readback present");
            println!(
                "[cublas readback] node {i}: source={:?} name={:?} handle={:p} \
                 grid=({}, {}, {}) block=({}, {}, {}) shared={} slots={}",
                rb.source,
                rb.name,
                rb.handle,
                rb.grid.x,
                rb.grid.y,
                rb.grid.z,
                rb.block.x,
                rb.block.y,
                rb.block.z,
                rb.shared,
                rb.cells.len()
            );
            assert!(!rb.handle.is_null(), "node {i}: kernel handle recovered");
            assert!(rb.grid.x > 0 && rb.block.x > 0, "node {i}: launch geometry recovered");
            match rb.source {
                ReadbackSource::Legacy => {
                    legacy_count += 1;
                    let (a, b, c) = &bufs[i];
                    let (ap, bp, cp) = (a.as_ptr() as u64, b.as_ptr() as u64, c.as_ptr() as u64);
                    println!(
                        "[cublas readback] node {i}: slot values = {:x?}\n  \
                         A={ap:x} B={bp:x} C={cp:x}",
                        &rb.values[..]
                    );
                    let layout = ParamLayout::Gemm { slots: 32, m: 0, n: 1, k: 2 };
                    let (rm, rn, rk) = rb
                        .gemm_mnk(&layout)
                        .unwrap_or_else(|| panic!("node {i}: m/n/k must be readable"));
                    // The cublas kernel packs the leading ints (m/n/k) as
                    // 4-byte halves of each 8-byte slot; `gemm_mnk` returns
                    // the full slot words, so compare low-32 only.
                    assert_eq!(
                        (rm as u32, rn as u32, rk as u32),
                        (m as u32, n as u32, k as u32),
                        "node {i}: geometry identification (m/n/k)"
                    );
                    // alpha = 1.0f (0x3f800000) must be present somewhere in
                    // the packed arg values.
                    assert!(
                        rb.values.iter().any(|&v| (v as u32) == 1.0f32.to_bits()),
                        "node {i}: alpha=1.0 readable"
                    );
                    // Operand pointers: empirically either the raw A/B/C
                    // addresses (direct) or cublas-staging addresses (the
                    // kernel reads through cublas-owned scratch). Both are
                    // recorded; direct recovery is asserted when it happens,
                    // staging is a conclusion for the engine wave.
                    let direct =
                        rb.values.iter().filter(|&&v| v == ap || v == bp || v == cp).count();
                    println!(
                        "[cublas readback] node {i}: m/n/k = {rm}/{rn}/{rk} identified, \
                         alpha readable, operand pointers {} ({direct}/3 direct)",
                        if direct == 3 { "DIRECT" } else { "via cublas staging" }
                    );
                    if direct != 3 {
                        values_ok = false;
                    }
                }
                ReadbackSource::V2 => {
                    println!(
                        "[cublas readback] node {i}: V2-only — argument *values* are not \
                         recoverable (capture-time cell addresses only); m/n/k geometry \
                         identification is not possible on this runtime"
                    );
                    values_ok = false;
                }
            }
        }
        println!(
            "[cublas readback] CONCLUSION: source={:?}, m/n/k+alpha+ptrs readable={values_ok} \
             (runtime {major}.{minor})",
            exec.node_params(0).expect("rb0").source
        );

        // Same-pointer replay: with no declared pointer slots the updates
        // are empty, so a replay is a plain launch of the captured kernel
        // args. On a Legacy read-back the staging was seeded from the
        // driver-owned values, so the launch sees the capture-time
        // arguments (m/n/k/alpha/operand staging) and must equal eager
        // bitwise. On a V2-only read-back the staging is unseeded — the
        // refresh-safety guard fails the replay closed (counted
        // `eager_fallback`), which is itself the asserted behavior.
        match exec.replay(&stream, &[]) {
            Ok(()) => {
                stream.synchronize().expect("sync");
                // Eager reference on the same buffers, then bitwise
                // compare each C.
                for (i, (m, n, k)) in shapes.into_iter().enumerate() {
                    let (a, b, c) = &bufs[i];
                    let c_bytes = m as usize * n as usize * 4;
                    let mut h = HostBuffer::alloc(c_bytes).expect("hC");
                    copy(&mut MemRef::Host(&mut h), &mut MemRef::Device(c), c_bytes, None)
                        .expect("d2h");
                    let got: Vec<f32> = unsafe {
                        core::slice::from_raw_parts(
                            h.as_ptr() as *const f32,
                            m as usize * n as usize,
                        )
                    }
                    .to_vec();
                    // Re-run eager over the same buffers.
                    let mut gm = gmat(c.as_ptr().cast_mut().cast(), f32, m);
                    gemm.gemm_f32acc(
                        &stream,
                        m,
                        n,
                        k,
                        &gmat(a.as_ptr().cast_mut().cast(), a16, k),
                        &gmat(b.as_ptr().cast_mut().cast(), a16, k),
                        &mut gm,
                        1.0,
                        0.0,
                    )
                    .expect("gemm eager");
                    stream.synchronize().expect("sync 2");
                    let mut h2 = HostBuffer::alloc(c_bytes).expect("hC2");
                    copy(&mut MemRef::Host(&mut h2), &mut MemRef::Device(c), c_bytes, None)
                        .expect("d2h");
                    let want: Vec<f32> = unsafe {
                        core::slice::from_raw_parts(
                            h2.as_ptr() as *const f32,
                            m as usize * n as usize,
                        )
                    }
                    .to_vec();
                    assert_eq!(got.len(), want.len());
                    for (j, (g, w)) in got.iter().zip(&want).enumerate() {
                        assert_eq!(g.to_bits(), w.to_bits(), "node {i} slot {j}");
                    }
                    println!(
                        "[cublas readback] node {i}: replay == eager bitwise \
                         ({} f32s)",
                        got.len()
                    );
                }
                assert!(
                    legacy_count == shapes.len(),
                    "a non-legacy read-back must never produce a launchable replay"
                );
            }
            Err(e) => {
                println!(
                    "[cublas readback] replay failed closed: {e:?} \
                     (engine falls back to eager; `eager_fallback` counted)"
                );
                // The fail-closed path is only correct when the read-back
                // left the staging unseeded (V2-only).
                assert_ne!(legacy_count, shapes.len(), "legacy read-back replay failed");
            }
        }
    }
}
