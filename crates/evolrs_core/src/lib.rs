pub mod device;
pub mod dtype;
pub mod nn;
pub mod ops;
pub mod shapes;
pub mod tensor;

pub fn is_available() -> bool {
    tch::Cuda::is_available()
}

pub use evolrs_macros::*;
