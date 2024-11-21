use std::{fmt::Debug, hash::Hash};

use evolrs_macros::crate_shape;

pub trait Shape: 'static + Debug + Clone + Copy + Send + Sync + PartialEq + Eq + Hash {
    type Shape: 'static + Debug + Clone + Copy + Send + Sync + PartialEq + Eq + Hash;
    const DIMS: i64;
    const NELEMS: usize;
    fn dims() -> &'static [i64];
}

crate_shape!(0);
crate_shape!(1);
crate_shape!(2);
crate_shape!(3);
crate_shape!(4);
crate_shape!(5);
crate_shape!(6);
crate_shape!(7);
crate_shape!(8);

#[cfg(test)]
mod tests {}
