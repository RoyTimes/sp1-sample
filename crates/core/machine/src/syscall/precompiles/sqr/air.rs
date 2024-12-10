use crate::{
    memory::{value_as_limbs, MemoryWriteCols},
    operations::field::field_op::FieldOpCols,
};

use crate::{
    air::MemoryAirBuilder,
    operations::field::range::FieldLtCols,
    utils::{limbs_from_prev_access, pad_rows_fixed, words_to_bytes_le},
};
use sp1_curves::edwards::WORDS_FIELD_ELEMENT;

use generic_array::GenericArray;
use num::{BigUint, Zero};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use sp1_core_executor::{
    events::{ByteRecord, FieldOperation, PrecompileEvent},
    syscalls::SyscallCode,
    ExecutionRecord, Program,
};
use sp1_curves::{
    edwards::ed25519::Ed25519BaseField,
    params::{FieldParameters, Limbs, NumLimbs, NumWords},
};
use sp1_derive::AlignedBorrow;
use sp1_stark::{
    air::{BaseAirBuilder, InteractionScope, MachineAir, SP1AirBuilder},
    MachineRecord,
};
use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

/// The number of columns in the Uint256MulCols.
const NUM_COLS: usize = size_of::<SqrCols<u8>>();

#[derive(Default)]
pub struct SqrChip;

impl SqrChip {
    pub const fn new() -> Self {
        Self
    }
}

type WordsFieldElement = <Ed25519BaseField as NumWords>::WordsFieldElement;

/// A set of columns for the Uint256Mul operation.
#[derive(Debug, Clone, AlignedBorrow)]
#[repr(C)]
pub struct SqrCols<T> {
    /// The shard number of the syscall.
    pub shard: T,

    /// The clock cycle of the syscall.
    pub clk: T,

    /// The nonce of the operation.
    pub nonce: T,

    /// The pointer to the first input.
    pub x_ptr: T,

    // Memory columns.
    // x_memory is written to with the result, which is why it is of type MemoryWriteCols.
    pub x_memory: GenericArray<MemoryWriteCols<T>, WordsFieldElement>,

    // Output values. We compute x.wrapping_mul(x).
    pub output: FieldOpCols<T, Ed25519BaseField>,

    pub output_range_check: FieldLtCols<T, Ed25519BaseField>,

    pub is_real: T,
}

impl<F: PrimeField32> MachineAir<F> for SqrChip {
    type Record = ExecutionRecord;
    type Program = Program;

    fn name(&self) -> String {
        "Sqr".to_string()
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        output: &mut ExecutionRecord,
    ) -> RowMajorMatrix<F> {
        // Generate the trace rows & corresponding records for each chunk of events concurrently.
        let rows_and_records = input
            .get_precompile_events(SyscallCode::SQR)
            .chunks(1)
            .map(|events| {
                let mut records = ExecutionRecord::default();
                let mut new_byte_lookup_events = Vec::new();

                let rows = events
                    .iter()
                    .map(|(_, event)| {
                        let event = if let PrecompileEvent::Sqr(event) = event {
                            event
                        } else {
                            unreachable!()
                        };
                        let mut row: [F; NUM_COLS] = [F::zero(); NUM_COLS];
                        let cols: &mut SqrCols<F> = row.as_mut_slice().borrow_mut();

                        let x = BigUint::from_bytes_le(&words_to_bytes_le::<32>(&event.x));

                        // Assign basic values to the columns.
                        cols.is_real = F::one();
                        cols.shard = F::from_canonical_u32(event.shard);
                        cols.clk = F::from_canonical_u32(event.clk);
                        cols.x_ptr = F::from_canonical_u32(event.x_ptr);

                        // Populate memory columns.
                        for i in 0..WORDS_FIELD_ELEMENT {
                            cols.x_memory[i]
                                .populate(event.x_memory_records[i], &mut new_byte_lookup_events);
                        }

                        // Populate the output column.
                        let result = cols.output.populate(
                            &mut new_byte_lookup_events,
                            event.shard,
                            &x,
                            &x,
                            FieldOperation::Mul,
                        );

                        cols.output_range_check.populate(
                            &mut new_byte_lookup_events,
                            event.shard,
                            &result,
                            &BigUint::from_bytes_le(Ed25519BaseField::MODULUS),
                        );

                        row
                    })
                    .collect::<Vec<_>>();
                records.add_byte_lookup_events(new_byte_lookup_events);
                (rows, records)
            })
            .collect::<Vec<_>>();

        //  Generate the trace rows for each event.
        let mut rows = Vec::new();
        for (row, mut record) in rows_and_records {
            rows.extend(row);
            output.append(&mut record);
        }

        pad_rows_fixed(
            &mut rows,
            || {
                let mut row: [F; NUM_COLS] = [F::zero(); NUM_COLS];
                let cols: &mut SqrCols<F> = row.as_mut_slice().borrow_mut();

                let x = BigUint::zero();
                cols.output.populate(&mut vec![], 0, &x, &x, FieldOperation::Mul);

                row
            },
            input.fixed_log2_rows::<F, _>(self),
        );

        // Convert the trace to a row major matrix.
        let mut trace =
            RowMajorMatrix::new(rows.into_iter().flatten().collect::<Vec<_>>(), NUM_COLS);

        // Write the nonces to the trace.
        for i in 0..trace.height() {
            let cols: &mut SqrCols<F> = trace.values[i * NUM_COLS..(i + 1) * NUM_COLS].borrow_mut();
            cols.nonce = F::from_canonical_usize(i);
        }

        trace
    }

    fn included(&self, shard: &Self::Record) -> bool {
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            !shard.get_precompile_events(SyscallCode::SQR).is_empty()
        }
    }
}

impl<F> BaseAir<F> for SqrChip {
    fn width(&self) -> usize {
        NUM_COLS
    }
}

impl<AB> Air<AB> for SqrChip
where
    AB: SP1AirBuilder,
    Limbs<AB::Var, <Ed25519BaseField as NumLimbs>::Limbs>: Copy,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &SqrCols<AB::Var> = (*local).borrow();
        let next = main.row_slice(1);
        let next: &SqrCols<AB::Var> = (*next).borrow();

        // Constrain the incrementing nonce.
        builder.when_first_row().assert_zero(local.nonce);
        builder.when_transition().assert_eq(local.nonce + AB::Expr::one(), next.nonce);

        let x_limbs = limbs_from_prev_access(&local.x_memory);
        // Evaluate the x * x multiplication
        local.output.eval(builder, &x_limbs, &x_limbs, FieldOperation::Mul, local.is_real);

        // Assert that the correct result is being written to x_memory.
        builder
            .when(local.is_real)
            .assert_all_eq(local.output.result, value_as_limbs(&local.x_memory));

        // Read and write x.
        builder.eval_memory_access_slice(
            local.shard,
            local.clk.into() + AB::Expr::one(),
            local.x_ptr,
            &local.x_memory,
            local.is_real,
        );

        // Receive the arguments.
        builder.receive_syscall(
            local.shard,
            local.clk,
            local.nonce,
            AB::F::from_canonical_u32(SyscallCode::SQR.syscall_id()),
            local.x_ptr,
            local.x_ptr,
            local.is_real,
            InteractionScope::Local,
        );

        // Assert that is_real is a boolean.
        builder.assert_bool(local.is_real);
    }
}