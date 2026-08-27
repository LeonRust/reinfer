//! 手动探针：cuLaunchKernel 流变体二分（012 C3 排查；feature `cuda`）。
//!
//! 用法（判定机）：
//! ```text
//! REINFER_CUDA_NVCC=/usr/local/cuda-12.8/bin/nvcc \
//! REINFER_JIT_CACHE=/tmp/reinfer-jit-probe \
//! cargo run -p reinfer-cuda --features cuda --example kernel_probe -- s1
//! ```
//! `s0`=NULL 流（legacy default）；`s1`=runtime 命名流（经 crate 的 cubin loader）。

use reinfer_core::DeviceId;
use reinfer_cuda::jit::{CtxGuard, JLib, launch_vec_add};
use reinfer_cuda::{CudaContext, CudaStream, DeviceBuffer, HostBuffer, MemRef, copy};

const N: u32 = 1 << 20;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let variant = std::env::args().nth(1).unwrap_or_else(|| "s0".into());
    println!("variant = {variant} (s0=NULL stream, s1=named stream)");

    let _ctx = CudaContext::init(DeviceId::new(0))?;
    // loadtest4 先显式 cuInit(0)（C3 二分：探针差异之一）
    let r = unsafe { cudarc::driver::sys::cuInit(0) };
    println!("[0] cuInit rc = {r:?}");
    let _guard = CtxGuard::set_current(0)?;
    println!("[1] runtime ctx + driver primary ctx set");

    let stream = if variant == "s1" { Some(CudaStream::new(DeviceId::new(0))?) } else { None };

    let dev = DeviceId::new(0);
    let ds = DeviceBuffer::alloc(dev, N as usize * 4)?;
    let db = DeviceBuffer::alloc(dev, N as usize * 4)?;
    let do_ = DeviceBuffer::alloc(dev, N as usize * 4)?;
    let hs = HostBuffer::alloc(N as usize * 4)?;
    // SAFETY: pinned host 内存由结构持有
    unsafe {
        let s = core::slice::from_raw_parts_mut(hs.as_ptr() as *mut f32, N as usize);
        for (i, x) in s.iter_mut().enumerate() {
            *x = i as f32;
        }
    }
    copy(&mut MemRef::Device(&ds), &MemRef::Host(&hs), N as usize * 4, None)?;
    println!("[2] buffers ready (4 MiB)");

    let tc = reinfer_jit::probe_toolchain()?;
    let arch = reinfer_cuda::arch::resolve_arch().expect("resolve arch");
    let src = reinfer_jit::KernelSource {
        name: "vec_add",
        src: include_str!("../kernels/vec_add.cu"),
        headers: vec![],
        flags: reinfer_jit::compile::gencode_flags(&arch)?,
        arch: arch.clone(),
        toolchain_ver: tc.ver_line.clone(),
    };
    let key = reinfer_jit::JitKey::new(&src, &tc);
    let cache = reinfer_jit::JitCache::open(None)?;
    let (_, p) = cache.build_once(&key, &src, || reinfer_jit::compile::compile_cubin(&src, &tc))?;
    let lib = JLib::from_bytes(std::fs::read(&p)?)?;
    let kernel = lib.kernel("vec_add")?;
    println!("[3] cubin loaded, kernel looked up");

    // s2：全 driver 面复刻 loadtest4（cuMemAlloc/cuMemcpy 全程 driver API）
    if variant == "s2" {
        let cfg = reinfer_core::DType::F32;
        let _ = cfg;
        let mut d_a: cudarc::driver::sys::CUdeviceptr = 0;
        let mut d_b: cudarc::driver::sys::CUdeviceptr = 0;
        let mut d_c: cudarc::driver::sys::CUdeviceptr = 0;
        let nbytes = N as usize * 4;
        // SAFETY: 输出槽位有效
        unsafe {
            let _ = cudarc::driver::sys::cuMemAlloc_v2(&mut d_a, nbytes);
            let _ = cudarc::driver::sys::cuMemAlloc_v2(&mut d_b, nbytes);
            let _ = cudarc::driver::sys::cuMemAlloc_v2(&mut d_c, nbytes);
        }
        let host: Vec<f32> = (0..N as usize).map(|i| i as f32 * 2.0).collect();
        println!("[s2] driver alloc ok, copying...");
        let r = unsafe {
            cudarc::driver::sys::cuMemcpyHtoD_v2(
                d_a,
                host.as_ptr().cast::<std::ffi::c_void>(),
                nbytes,
            )
        };
        println!("[s2] h2d rc = {r:?}");
        let n = 0i32; // dummy
        let _ = n;
        let mut args: [*mut std::ffi::c_void; 4] = [
            (&d_a as *const _) as *mut std::ffi::c_void,
            (&d_b as *const _) as *mut std::ffi::c_void,
            (&d_c as *const _) as *mut std::ffi::c_void,
            (&N as *const u32).cast::<std::ffi::c_void>() as *mut std::ffi::c_void,
        ];
        println!("[s2] launching driver-path...");
        let r = unsafe {
            cudarc::driver::sys::cuLaunchKernel(
                kernel.raw(),
                N.div_ceil(256),
                1,
                1,
                256,
                1,
                1,
                0,
                std::ptr::null_mut(),
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        println!("[s2] launch rc = {r:?}");
        let _ = &mut args;
        let _ = &d_b;
        return Ok(());
    }

    // s3：runtime 分配 + driver launch（二分：内存面 vs launch 面）
    if variant == "s3" {
        let mut args: [*mut std::ffi::c_void; 4] = [
            ds.as_ptr() as cudarc::driver::sys::CUdeviceptr as *mut std::ffi::c_void,
            db.as_ptr().cast::<std::ffi::c_void>() as *mut std::ffi::c_void,
            do_.as_ptr().cast::<std::ffi::c_void>() as *mut std::ffi::c_void,
            (&N as *const u32).cast::<std::ffi::c_void>() as *mut std::ffi::c_void,
        ];
        println!("[s3] launching with runtime-alloc pointers...");
        let r = unsafe {
            cudarc::driver::sys::cuLaunchKernel(
                kernel.raw(),
                N.div_ceil(256),
                1,
                1,
                256,
                1,
                1,
                0,
                std::ptr::null_mut(),
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        println!("[s3] launch rc = {r:?}");
        return Ok(());
    }

    // s4：指针归属诊断（runtime 指针的 context vs primary retain context）
    if variant == "s4" {
        let mut ctx_out: *mut std::ffi::c_void = std::ptr::null_mut();
        let r = unsafe {
            cudarc::driver::sys::cuPointerGetAttribute(
                std::ptr::addr_of_mut!(ctx_out).cast::<std::ffi::c_void>(),
                cudarc::driver::sys::CUpointer_attribute::CU_POINTER_ATTRIBUTE_CONTEXT,
                ds.as_ptr() as cudarc::driver::sys::CUdeviceptr,
            )
        };
        println!("[s4] runtime ptr context attr rc={r:?} ctx={ctx_out:?}");
        // primary retain 的 ctx：
        let mut pctx: cudarc::driver::sys::CUcontext = std::ptr::null_mut();
        let r2 = unsafe { cudarc::driver::sys::cuDevicePrimaryCtxRetain(&mut pctx, 0) };
        println!("[s4] retain rc={r2:?} pctx={pctx:?}");
        println!("[s4] same? {}", std::ptr::eq(ctx_out, pctx as *const std::ffi::c_void));
        return Ok(());
    }

    // s5：runtime 分配 + s2 同款参数写法（CUdeviceptr 变量取址）
    if variant == "s5" {
        let d_a: cudarc::driver::sys::CUdeviceptr = ds.as_ptr() as cudarc::driver::sys::CUdeviceptr;
        let d_b: cudarc::driver::sys::CUdeviceptr = db.as_ptr() as cudarc::driver::sys::CUdeviceptr;
        let d_c: cudarc::driver::sys::CUdeviceptr =
            do_.as_ptr() as cudarc::driver::sys::CUdeviceptr;
        let n = N;
        let p_a = &d_a as *const cudarc::driver::sys::CUdeviceptr as *mut std::ffi::c_void;
        let p_b = &d_b as *const cudarc::driver::sys::CUdeviceptr as *mut std::ffi::c_void;
        let p_c = &d_c as *const cudarc::driver::sys::CUdeviceptr as *mut std::ffi::c_void;
        let p_n = &n as *const u32 as *mut std::ffi::c_void;
        let mut args: [*mut std::ffi::c_void; 4] = [p_a, p_b, p_c, p_n];
        println!("[s5] args like s2, runtime-alloc pointers");
        let r = unsafe {
            cudarc::driver::sys::cuLaunchKernel(
                kernel.raw(),
                N.div_ceil(256),
                1,
                1,
                256,
                1,
                1,
                0,
                std::ptr::null_mut(),
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        println!("[s5] launch rc = {r:?}");
        return Ok(());
    }

    println!("[4] launching (variant={variant})...");
    std::io::Write::flush(&mut std::io::stdout())?;
    match &stream {
        Some(s) => {
            // SAFETY: 探针内指针有效；s1 分支与真机测试同路径
            unsafe {
                launch_vec_add(
                    kernel,
                    s,
                    0,
                    ds.as_ptr().cast::<f32>(),
                    db.as_ptr().cast::<f32>(),
                    do_.as_ptr().cast::<f32>() as *mut f32,
                    N,
                )
            }?;
            s.synchronize()?;
        }
        None => {
            let mut args: [*mut std::ffi::c_void; 4] = [
                ds.as_ptr().cast::<std::ffi::c_void>() as *mut std::ffi::c_void,
                db.as_ptr().cast::<std::ffi::c_void>() as *mut std::ffi::c_void,
                do_.as_ptr().cast::<std::ffi::c_void>() as *mut std::ffi::c_void,
                (&N as *const u32).cast::<std::ffi::c_void>() as *mut std::ffi::c_void,
            ];
            // SAFETY: 同 launch_vec_add 前提
            let r = unsafe {
                cudarc::driver::sys::cuLaunchKernel(
                    kernel.raw(),
                    N.div_ceil(256),
                    1,
                    1,
                    256,
                    1,
                    1,
                    0,
                    std::ptr::null_mut(),
                    args.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            };
            if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                println!("[5] launch rc = {r:?}");
                return Ok(());
            }
            unsafe { cudarc::runtime::sys::cudaDeviceSynchronize() }
                .result()
                .map_err(|e| format!("cudaDeviceSynchronize: {e:?}"))?;
        }
    }
    println!("[5] launch ok, device synced");

    let hout = HostBuffer::alloc(N as usize * 4)?;
    copy(&mut MemRef::Host(&hout), &MemRef::Device(&do_), N as usize * 4, None)?;
    // SAFETY: 只读
    let got = unsafe { core::slice::from_raw_parts(hout.as_ptr() as *const f32, N as usize) };
    let mut ok = true;
    for (i, &g) in got.iter().enumerate() {
        if (g - 2.0 * i as f32).abs() > 1e-4 {
            ok = false;
            println!("mismatch at {i}: {g} vs {}", 2.0 * i as f32);
            break;
        }
    }
    println!("[6] result check: {ok}");
    let _ = &db;
    Ok(())
}
