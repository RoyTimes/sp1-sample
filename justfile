test-opcode:
    cargo test --package sp1-core-executor --lib -- executor::tests::square_tests --exact --show-output

test-sqr-chip:
    cargo test --package sp1-core-machine --lib -- alu::sqr::tests --show-output

test:
    just test-opcode
    just test-sqr-chip
