//! One-off probe: SM count / max threads per SM / max blocks per SM for a
//! 512-thread block (co-residency math for the S1-10 fused layer kernel).
#![cfg(feature = "cuda")]
use cudarc::driver::sys::{self, CUdevice_attribute, CUresult};

fn attr(dev: sys::CUdevice, a: CUdevice_attribute) -> i32 {
    let mut v: i32 = 0;
    let r = unsafe { sys::cuDeviceGetAttribute(&mut v, a, dev) };
    assert_eq!(r, CUresult::CUDA_SUCCESS, "attr {a:?}");
    v
}

fn main() {
    let mut count: i32 = 0;
    unsafe {
        let _ = sys::cuInit(0);
        let r = sys::cuDeviceGetCount(&mut count);
        assert_eq!(r, CUresult::CUDA_SUCCESS);
    }
    for i in 0..count {
        let dev: sys::CUdevice = i;
        let sms = attr(dev, sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT);
        let mthr =
            attr(dev, sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR);
        let mthr_bt = attr(dev, sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
        let mblk =
            attr(dev, sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR);
        let shmem = attr(
            dev,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
        );
        let coop = attr(dev, sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH);
        println!(
            "dev{i}: sms={sms} max_threads_per_sm={mthr} max_threads_per_block={mthr_bt} \
             max_blocks_per_sm={mblk} smem_per_sm={shmem} cooperative_launch={coop}"
        );
        let block = 512;
        println!("  => resident blocks (512 thr, no smem): {}", (mthr / block) * sms);
    }
}
