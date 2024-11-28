use crate::{
    device::Device,
    kind::Kind,
    shapes::shape::{Rank1, Shape},
    tensor::{RequiresGrad, Tensor},
};

pub trait Flatten<S: Shape>: Shape {
    const CHECK: ();
    fn comptime_check() {
        #![allow(path_statements)]
        Self::CHECK;
    }
}

impl<const NELEMS: usize, S: Shape> Flatten<S> for Rank1<NELEMS> {
    const CHECK: () = assert!(
        NELEMS == S::NELEMS,
        "Flatten shape must have the same number of elements as the original shape"
    );
}

impl<S: Shape, D: Device, K: Kind, G: RequiresGrad> Tensor<S, D, K, G> {
    pub fn flatten<N: Flatten<S>>(&self) -> Tensor<N, D, K> {
        N::comptime_check();
        Tensor {
            repr: self.repr.flatten(0, -1),
            ..Default::default()
        }
    }

    pub fn flatten_n<const N: usize>(&self) -> Tensor<Rank1<N>, D, K> {
        <Rank1<N> as Flatten<S>>::comptime_check();
        Tensor {
            repr: self.repr.flatten(0, -1),
            ..Default::default()
        }
    }
}

///```compile_fail
/// use crate::evolrs_core::{shapes::shape::{ Rank3, Rank1 }, tensor::Tensor};
/// let t1: Tensor<Rank3<1, 2, 3>> = Tensor::ones();
/// let _ = t1.flatten::<Rank1<7>>();
/// ```
///
/// ```compile_fail
/// use crate::evolrs_core::{shapes::shape::{ Rank3, Rank1 }, tensor::Tensor};
/// let t1: Tensor<Rank3<1, 2, 3>> = Tensor::ones();
/// let _ = t1.flatten_n::<7>();
/// ```
mod comptime_fails_flatten {}

#[cfg(test)]
mod tests {}
