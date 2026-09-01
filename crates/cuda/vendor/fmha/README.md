# vendored FMHA headers (flash-attn v2.8.3 kernel closure)

Batched-prefill FMHA kernel source for `crates/cuda/src/fmha.rs`, per
specs/006 D1 (`Vendor(cubin) > Jit(fmha) > Jit(dense)`).

## Layout

- `headers/` — the header closure consumed at JIT compile time:
  - 11 flash-attn files at the root (flash_fwd_kernel.h and its direct
    dependencies), license headers kept as upstream;
  - `cute/` and `cutlass/` — transitive include closure from the cutlass
    submodule pinned by flash-attn v2.8.3 (only headers reachable from
    flash_fwd_kernel.h; no full cutlass tree);
  - `philox_unpack.cuh` — **patched**: upstream's file needs
    `<ATen/cuda/detail/UnpackRaw.cuh>` (torch); reinfer has no torch, so
    this header defines a minimal `at::cuda::philox::PhiloxCudaState`
    (seed + offset) and `unpack()` overload. Dropout is disabled at
    compile time (`Is_dropout=false`), so the unpacked values are never
    used; the call site in flash_fwd_kernel.h is unconditional, hence the
    shim. See version.json `patched_files`.
- `version.json` — source repos, pinned commits, extracted file list,
  extraction date. Bumping `version` invalidates JitCache keys (headers
  are hashed into `JitKey`).
- `extract.py` — reproduces the extraction: `python3 extract.py
  <flash-attn-upstream> headers/` (upstream = v2.8.3 clone with
  submodules checked out; cutlass lives at `csrc/cutlass/include`).
- `upstream/` — scratch clone used for the extraction (426 MB incl.
  submodules); **not** part of the vendored artifact. Keep it out of
  builds and out of the repo if it is not committed.

## Consumption (lookup path)

`crates/cuda/src/fmha.rs` reads the headers at runtime from
`<CARGO_MANIFEST_DIR>/vendor/fmha/headers/` and passes them to
`JitCache` as `HeaderFile { path, content }` — the JitKey hashes the
content, so mtime is irrelevant and any touch of this tree invalidates
the cache. The build must therefore be able to resolve that path at
runtime on the machine that JIT-compiles; it is a checked-in directory.

## Kernel design (one line)

Own `extern "C"` wrapper in `crates/cuda/kernels/fmha_kernels.cu`
instantiates
`flash::compute_attn<Flash_fwd_kernel_traits<128,128,128,4,false,false,cutlass::half_t>, Is_dropout=false, Is_causal=true, Is_local=false, Has_alibi=false, Is_even_MN, Is_even_K=true, Is_softcap=false, Return_softmax=false>`
(SM80_16x8x16_F32F16F16F32_TN MMA atoms — the same atom family for all
archs >= 800, so no CUTE arch flags are needed and sm_120a works), fed
with contiguous [S,B,nqk] buffers via affine strides (no transposes),
GQA via h_k param, causal mask, rotary_dim=0 (engine pre-applies RoPE),
q pre-scaled by 1/sqrt(d) with scale_softmax=1.0 (matches the per-token
decode path bit-for-bit for the score scale).

## License

Both upstream projects are BSD-3-Clause; see `LICENSE`.
