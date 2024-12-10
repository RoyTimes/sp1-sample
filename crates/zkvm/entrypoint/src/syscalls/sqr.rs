#[cfg(target_os = "zkvm")]
use core::arch::asm;

const N: usize = 1;
/// Sets `result` to be `x^2`.
#[allow(unused_variables)]
#[no_mangle]
pub extern "C" fn syscall_sqr(result: *mut [u32; N], x: *const [u32; N]) {
    // Instantiate a new uninitialized array of words to place the concatenated y and modulus.
    unsafe {
        let result_ptr = result as *mut u32;
        let x_ptr = x as *const u32;

        // Copy x into the result array, as our syscall will write the result into the first input.
        core::ptr::copy(x_ptr, result_ptr, N);
        let result_ptr = result_ptr as *mut [u32; N];

        #[cfg(target_os = "zkvm")]
        unsafe {
            asm!(
                "ecall",
                in("t0") crate::syscalls::SQR,
                in("a0") result_ptr,
                in("a1") x_ptr, // dummy value
            );
        }

        #[cfg(not(target_os = "zkvm"))]
        unreachable!()
    }
}
