pub use evolrs_core::*;

#[test]
fn comptime_fail_tests() {
    trybuild::TestCases::new().compile_fail("tests/comptime_fail/*/*.rs");
}
