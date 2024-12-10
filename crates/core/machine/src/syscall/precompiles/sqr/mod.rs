mod air;

pub use air::*;

#[cfg(test)]
mod tests {

    use sp1_core_executor::Program;
    use sp1_stark::CpuProver;
    use test_artifacts::SQR_ELF;

    use crate::{
        io::SP1Stdin,
        utils::{self, run_test_io},
    };

    #[test]
    fn test_sqr() {
        utils::setup_logger();
        let program = Program::from(SQR_ELF).unwrap();
        run_test_io::<CpuProver<_, _>>(program, SP1Stdin::new()).unwrap();
    }
}
