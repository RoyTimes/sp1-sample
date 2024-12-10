#![no_main]

sp1_zkvm::entrypoint!(main);

use num_bigint::BigUint;
use sp1_curves::edwards::WORDS_FIELD_ELEMENT;
use sp1_zkvm::syscalls::syscall_sqr;

fn sqr(x: u32) -> u64 {
    let x_bigint = BigUint::from_u32(x).unwrap();
    let x_bigint_to_bytes = x_bigint.to_bytes_le();
    let x_words = bytemuck::cast::<[u8; 32], [u32; 8]>(x_bigint_to_bytes);
    let result = sqr_inner(x_words);
    let result_bytes = bytemuck::cast::<[u32; 8], [u8; 32]>(result);
    let result_bigint = BigUint::from_bytes_le(&result_bytes);
    result_bigint.to_u64().unwrap()
}

fn sqr_inner(x: [u32; WORDS_FIELD_ELEMENT]) -> [u32; WORDS_FIELD_ELEMENT] {
    let mut result = [0u32; WORDS_FIELD_ELEMENT];
    syscall_sqr(
        result.as_mut_ptr() as *mut [u32; WORDS_FIELD_ELEMENT],
        x.as_ptr() as *const [u32; WORDS_FIELD_ELEMENT],
    );
    result
}

pub fn main() {
    let result_1 = sqr(1u32);
    assert_eq!(result_1, 1u64);

    let result2 = sqr(0u32);
    assert_eq!(result2, 0u64);

    let result3 = sqr(42u32);
    assert_eq!(result3, 1764u64);

    let result4 = sqr(0x12345678u32);
    assert_eq!(result4, 0x014B66DC_1DF4D840u64);

    println!("All tests passed!");
}
