#![allow(private_bounds)]
#![allow(path_statements)]

pub mod data;
pub mod device;
pub mod kind;
pub mod nn;
pub mod ops;
pub mod shapes;
pub mod tensor;
pub mod prelude {}

pub(crate) mod utils;

pub use evolrs_macros::*;
pub use tch;
