# Install Just to run this script: https://github.com/casey/just

test-opcode:
    RUST_LOG="debug" cargo test --package sp1-core-executor --lib -- executor::tests::square_tests --exact --show-output

test-sqr-chip:
    RUST_LOG="debug" cargo test --package sp1-core-machine --lib -- alu::sqr::tests --show-output

test-sqr-precompile:
    RUST_LOG="debug" cargo test --package sp1-core-machine --lib -- syscall::precompiles::sqr::tests::test_sqr --exact --show-output

test-precompile-consistency:
    RUST_LOG="debug" cargo test --package sp1-core-machine --lib -- runtime::syscall::tests --exact --show-output

test:
    just test-opcode
    just test-sqr-chip
    just test-sqr-precompile
