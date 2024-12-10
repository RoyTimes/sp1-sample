//! Implementation to check that b * c = product.
//!
//! We first extend the operands to 64 bits. We sign-extend them if the op code is signed. Then we
//! calculate the un-carried product and propagate the carry. Finally, we check that the appropriate
//! bits of the product match the result.
//!
//! b_64 = sign_extend(b) if signed operation else b
//! c_64 = sign_extend(c) if signed operation else c
//!
//! m = []
//! # 64-bit integers have 8 limbs.
//! # Calculate un-carried product.
//! for i in 0..8:
//!     for j in 0..8:
//!         if i + j < 8:
//!             m[i + j] += b_64[i] * c_64[j]
//!
//! # Propagate carry
//! for i in 0..8:
//!     x = m[i]
//!     if i > 0:
//!         x += carry[i - 1]
//!     carry[i] = x / 256
//!     m[i] = x % 256
//!
//! if upper_half:
//!     assert_eq(a, m[4..8])
//! if lower_half:
//!     assert_eq(a, m[0..4])

use core::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use hashbrown::HashMap;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, PrimeField};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::{ParallelBridge, ParallelIterator, ParallelSlice};
use sp1_core_executor::{
    events::{AluEvent, ByteLookupEvent, ByteRecord},
    ExecutionRecord, Opcode, Program,
};
use sp1_derive::AlignedBorrow;
use sp1_primitives::consts::WORD_SIZE;
use sp1_stark::{air::MachineAir, Word};

use crate::{
    air::SP1CoreAirBuilder,
    utils::{next_power_of_two, zeroed_f_vec},
};

/// The number of main trace columns for `MulChip`.
pub const NUM_SQR_COLS: usize = size_of::<SqrCols<u8>>();

/// The number of digits in the product is at most the sum of the number of digits in the
/// multiplicands.
const PRODUCT_SIZE: usize = 2 * WORD_SIZE;

/// The number of bits in a byte.
const BYTE_SIZE: usize = 8;

/// A chip that implements multiplication for the multiplication opcodes.
#[derive(Default)]
pub struct SqrChip;

/// The column layout for the chip.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct SqrCols<T> {
    /// The shard number, used for byte lookup table.
    pub shard: T,

    /// The nonce of the operation.
    pub nonce: T,

    /// The output operand.
    pub a: Word<T>,

    /// The first input operand.
    pub b: Word<T>,

    // NOTE: this can be ignored
    /// The second input operand.
    pub c: Word<T>,

    /// Trace.
    pub carry: [T; PRODUCT_SIZE],

    /// An array storing the product of `b * c` after the carry propagation.
    pub product: [T; PRODUCT_SIZE],

    /// Flag indicating whether the opcode is `SQR`.
    pub is_sqr: T,

    /// Selector to know whether this row is enabled.
    pub is_real: T,
}

impl<F: PrimeField> MachineAir<F> for SqrChip {
    type Record = ExecutionRecord;

    type Program = Program;

    fn name(&self) -> String {
        "Sqr".to_string()
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _: &mut ExecutionRecord,
    ) -> RowMajorMatrix<F> {
        // Generate the trace rows for each event.
        let nb_rows = input.mul_events.len();
        let size_log2 = input.fixed_log2_rows::<F, _>(self);
        let padded_nb_rows = next_power_of_two(nb_rows, size_log2);
        let mut values = zeroed_f_vec(padded_nb_rows * NUM_SQR_COLS);
        let chunk_size = std::cmp::max((nb_rows + 1) / num_cpus::get(), 1);

        values.chunks_mut(chunk_size * NUM_SQR_COLS).enumerate().par_bridge().for_each(
            |(i, rows)| {
                rows.chunks_mut(NUM_SQR_COLS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut SqrCols<F> = row.borrow_mut();

                    if idx < nb_rows {
                        let mut byte_lookup_events = Vec::new();
                        let event = &input.mul_events[idx];
                        self.event_to_row(event, cols, &mut byte_lookup_events);
                    }
                    cols.nonce = F::from_canonical_usize(idx);
                });
            },
        );

        // Convert the trace to a row major matrix.

        RowMajorMatrix::new(values, NUM_SQR_COLS)
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        let chunk_size = std::cmp::max(input.mul_events.len() / num_cpus::get(), 1);

        let blu_batches = input
            .mul_events
            .par_chunks(chunk_size)
            .map(|events| {
                let mut blu: HashMap<u32, HashMap<ByteLookupEvent, usize>> = HashMap::new();
                events.iter().for_each(|event| {
                    let mut row = [F::zero(); NUM_SQR_COLS];
                    let cols: &mut SqrCols<F> = row.as_mut_slice().borrow_mut();
                    self.event_to_row(event, cols, &mut blu);
                });
                blu
            })
            .collect::<Vec<_>>();

        output.add_sharded_byte_lookup_events(blu_batches.iter().collect::<Vec<_>>());
    }

    fn included(&self, shard: &Self::Record) -> bool {
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            !shard.mul_events.is_empty()
        }
    }
}

// NOTE: this is the main chip logic implementation
impl SqrChip {
    /// Create a row from an event.
    fn event_to_row<F: PrimeField>(
        &self,
        event: &AluEvent,
        cols: &mut SqrCols<F>,
        blu: &mut impl ByteRecord,
    ) {
        let a_word = event.a.to_le_bytes();
        let b_word = event.b.to_le_bytes();
        let c_word = event.c.to_le_bytes();

        let b = b_word.to_vec();
        let mut product = [0u32; PRODUCT_SIZE];
        for i in 0..b.len() {
            for j in 0..b.len() {
                if i + j < PRODUCT_SIZE {
                    product[i + j] += (b[i] as u32) * (b[j] as u32);
                }
            }
        }

        // Calculate the correct product using the `product` array. We store the
        // correct carry value for verification.
        let base = (1 << BYTE_SIZE) as u32;
        let mut carry = [0u32; PRODUCT_SIZE];
        for i in 0..PRODUCT_SIZE {
            carry[i] = product[i] / base;
            product[i] %= base;
            if i + 1 < PRODUCT_SIZE {
                product[i + 1] += carry[i];
            }
            cols.carry[i] = F::from_canonical_u32(carry[i]);
        }

        cols.product = product.map(F::from_canonical_u32);
        cols.a = Word(a_word.map(F::from_canonical_u8));
        cols.b = Word(b_word.map(F::from_canonical_u8));
        cols.c = Word(c_word.map(F::from_canonical_u8)); // we are still keeping the useless C here
        cols.is_real = F::one();
        cols.is_sqr = F::from_bool(event.opcode == Opcode::SQR);
        cols.shard = F::from_canonical_u32(event.shard);

        // Range check.
        {
            blu.add_u16_range_checks(event.shard, &carry.map(|x| x as u16));
            blu.add_u8_range_checks(event.shard, &product.map(|x| x as u8));
        }
    }
}

impl<F> BaseAir<F> for SqrChip {
    fn width(&self) -> usize {
        NUM_SQR_COLS
    }
}

impl<AB> Air<AB> for SqrChip
where
    AB: SP1CoreAirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &SqrCols<AB::Var> = (*local).borrow();
        let next = main.row_slice(1);
        let next: &SqrCols<AB::Var> = (*next).borrow();
        let base = AB::F::from_canonical_u32(1 << 8);

        // Constrain the incrementing nonce.
        builder.when_first_row().assert_zero(local.nonce);
        builder.when_transition().assert_eq(local.nonce + AB::Expr::one(), next.nonce);

        let mut b: Vec<AB::Expr> = vec![AB::F::zero().into(); PRODUCT_SIZE];
        for i in 0..PRODUCT_SIZE {
            if i < WORD_SIZE {
                b[i] = local.b[i].into();
            }
        }

        // Compute the uncarried product b(x) * c(x) = m(x).
        let mut m: Vec<AB::Expr> = vec![AB::F::zero().into(); PRODUCT_SIZE];
        for i in 0..PRODUCT_SIZE {
            for j in 0..PRODUCT_SIZE {
                if i + j < PRODUCT_SIZE {
                    m[i + j] = m[i + j].clone() + b[i].clone() * b[j].clone();
                }
            }
        }

        // Propagate carry.
        let product = {
            for i in 0..PRODUCT_SIZE {
                if i == 0 {
                    builder.assert_eq(local.product[i], m[i].clone() - local.carry[i] * base);
                } else {
                    builder.assert_eq(
                        local.product[i],
                        m[i].clone() + local.carry[i - 1] - local.carry[i] * base,
                    );
                }
            }
            local.product
        };
        for i in 0..WORD_SIZE {
            // ONLY compare the lower half
            builder.assert_eq(product[i], local.a[i]);
        }

        // Check that the boolean values are indeed boolean values.
        {
            let booleans = [local.is_sqr, local.is_real];
            for boolean in booleans.iter() {
                builder.assert_bool(*boolean);
            }
        }

        // Calculate the opcode.
        let opcode = {
            // Exactly one of the op codes must be on.
            builder.when(local.is_real).assert_one(local.is_sqr); // NOTE: this might be trivial

            let sqr: AB::Expr = AB::F::from_canonical_u32(Opcode::SQR as u32).into();
            sqr
        };

        // Range check.
        {
            // Ensure that the carry is at most 2^16. This ensures that
            // product_before_carry_propagation - carry * base + last_carry never overflows or
            // underflows enough to "wrap" around to create a second solution.
            builder.slice_range_check_u16(&local.carry, local.is_real);

            builder.slice_range_check_u8(&local.product, local.is_real);
        }

        // Receive the arguments.
        builder.receive_alu(
            opcode,
            local.a,
            local.b,
            local.c,
            local.shard,
            local.nonce,
            local.is_real,
        );
    }
}

#[cfg(test)]
mod tests {

    use crate::utils::{uni_stark_prove as prove, uni_stark_verify as verify};
    use p3_baby_bear::BabyBear;
    use p3_matrix::dense::RowMajorMatrix;
    use sp1_core_executor::{events::AluEvent, ExecutionRecord, Opcode};
    use sp1_stark::{air::MachineAir, baby_bear_poseidon2::BabyBearPoseidon2, StarkGenericConfig};

    use super::SqrChip;

    #[test]
    fn generate_trace_mul() {
        let mut shard = ExecutionRecord::default();

        // Fill mul_events with 10^7 MULHSU events.
        let mul_events: Vec<AluEvent> = vec![
            AluEvent::new(0, 0, Opcode::SQR, 0x06E4, 0x2A, 0x1234),
            AluEvent::new(0, 0, Opcode::SQR, 0x1df4d840, 0x12345678, 0x1234),
        ];
        shard.mul_events = mul_events;
        let chip = SqrChip::default();
        let _trace: RowMajorMatrix<BabyBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default());
    }

    #[test]
    fn prove_sqr_babybear() {
        // NOTE: this prover proves 42 * 42 = 1764 | 0x2A * 0x2A = 0x06E4
        // 0x12345678 * 0x12345678 = 0x1df4d840

        let config = BabyBearPoseidon2::new();
        let mut challenger = config.challenger();

        let mut shard = ExecutionRecord::default();
        let mut mul_events: Vec<AluEvent> = Vec::new();

        let mul_instructions: Vec<(Opcode, u32, u32, u32)> = vec![
            (Opcode::SQR, 0x0, 0x0, 0x00),
            (Opcode::SQR, 0x1, 0x1, 0x00),
            (Opcode::SQR, 0x06E4, 0x2A, 0x00),
            (Opcode::SQR, 0x1df4d840, 0x12345678, 0x00),
        ];
        for t in mul_instructions.iter() {
            mul_events.push(AluEvent::new(0, 0, t.0, t.1, t.2, t.3));
        }

        shard.mul_events = mul_events;
        let chip = SqrChip::default();
        let trace: RowMajorMatrix<BabyBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default());
        // println!("{:?}", trace.values); // out of curiosity
        let proof = prove::<BabyBearPoseidon2, _>(&config, &chip, &mut challenger, trace);

        let mut challenger = config.challenger();
        verify(&config, &chip, &mut challenger, &proof).unwrap();
    }
}
