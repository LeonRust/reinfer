// reinfer-compat shim for `at::cuda::philox` (see vendor/fmha/README.md).
//
// Upstream flash-attn (csrc/flash_attn/src/philox_unpack.cuh) unpacks the
// torch `at::PhiloxCudaState` via <ATen/cuda/detail/UnpackRaw.cuh>. reinfer
// has no torch dependency, so this header defines a minimal stand-in state
// type and the `unpack` overload that `flash_fwd_kernel.h` calls
// unconditionally at kernel entry (dropout is disabled at compile time, so
// the unpacked values are never used).
//
// The reinfer params struct (kernels/fmha_kernels.cu) declares its
// `philox_args` field with this type; flash_fwd_kernel.h's call
// `at::cuda::philox::unpack(params.philox_args)` resolves to the overload
// below. Kept in this header tree so that JitCache header hashing sees it
// like any other vendored header.

#pragma once

#include <tuple>

namespace at::cuda::philox {

/// Stand-in for torch `at::PhiloxCudaState` (seed + per-launch offset).
struct PhiloxCudaState {
  unsigned long long seed_;
  unsigned long long offset_;
};

/// Match the upstream signature: returns a (seed, offset) tuple.
__forceinline__ __device__ std::tuple<unsigned long long, unsigned long long> unpack(
    const PhiloxCudaState& s) {
  return std::make_tuple(s.seed_, s.offset_);
}

}  // namespace at::cuda::philox
