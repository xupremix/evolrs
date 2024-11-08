use std::{fmt::Debug, hash::Hash};

use evolrs_macros::crate_shape;

pub trait Shape: 'static + Debug + Clone + Copy + Send + Sync + PartialEq + Eq + Hash {
    type Shape: 'static + Debug + Clone + Copy + Send + Sync + PartialEq + Eq + Hash;
    const DIMS: usize;
    const NELEMS: usize;
    fn dims() -> &'static [i64];
}

crate_shape!(pub 0);
crate_shape!(pub 1);
crate_shape!(pub 2);
crate_shape!(pub 3);
crate_shape!(pub 4);
crate_shape!(pub 5);
crate_shape!(pub 6);
crate_shape!(pub 7);
crate_shape!(pub 8);

#[cfg(test)]
mod tests {}
