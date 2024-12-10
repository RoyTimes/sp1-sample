use num::BigUint;

use sp1_curves::edwards::WORDS_FIELD_ELEMENT;
use sp1_primitives::consts::{bytes_to_words_le, words_to_bytes_le_vec};

use crate::{
    events::{PrecompileEvent, SqrEvent},
    syscalls::{Syscall, SyscallCode, SyscallContext},
};

// NOTE: this is inspired by the uint256_mul syscall.
// As uint256_mul is impl over edwards, we keep it this way
// We aim to maintain the same behavior as SQR Opcode - only keep the lower parts
pub(crate) struct SqrSyscall;

impl Syscall for SqrSyscall {
    fn execute(
        &self,
        rt: &mut SyscallContext,
        syscall_code: SyscallCode,
        arg1: u32,
        _arg2: u32, // we keep ths dummy value for consistency with the uint256_mul syscall & SQR opcode.
    ) -> Option<u32> {
        let clk = rt.clk;

        let x_ptr = arg1;
        if x_ptr % 4 != 0 {
            panic!();
        }

        // First read the words for the x value. We can read a slice_unsafe here because we write
        // the computed result to x later.

        // NOTE: don't wanna dig into the Curve impl, so we all use BigUint here
        let x_words = rt.slice_unsafe(x_ptr, WORDS_FIELD_ELEMENT);
        let x_bigint = BigUint::from_bytes_le(&words_to_bytes_le_vec(&x_words));

        let result = x_bigint.pow(2);
        let mut result_bytes = result.to_bytes_le();

        // we want the result in u32 - therefore, resize to 4
        result_bytes.resize(WORDS_FIELD_ELEMENT * 4, 0u8);

        // Convert the result to little endian u32 words.
        let result = bytes_to_words_le::<WORDS_FIELD_ELEMENT>(&result_bytes);

        // Increment clk so that the write is not at the same cycle as the read.
        rt.clk += 1;
        // Write the result to x and keep track of the memory records.
        let x_memory_records = rt.mw_slice(x_ptr, &result);

        let lookup_id = rt.syscall_lookup_id;
        let shard = rt.current_shard();
        let event = PrecompileEvent::Sqr(SqrEvent {
            lookup_id,
            shard,
            clk,

            x_ptr,
            x: x_words,
            x_memory_records,
            local_mem_access: rt.postprocess(),
        });
        let sycall_event =
            rt.rt.syscall_event(clk, syscall_code.syscall_id(), arg1, _arg2, lookup_id);
        rt.add_precompile_event(syscall_code, sycall_event, event);

        None
    }

    fn num_extra_cycles(&self) -> u32 {
        1
    }
}
