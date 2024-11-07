pub mod device;
pub mod kind;
pub mod nn;
pub mod ops;
pub mod shapes;
pub mod tensor;

pub fn e() {
    let a = tch::Tensor::from_slice(&[1, 2, 3]);
    let a = a / 2;

    println!("{:?}", a.kind());
}

pub use evolrs_macros::*;
