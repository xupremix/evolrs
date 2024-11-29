use std::marker::PhantomData;

pub mod gen;
pub mod wrap;

use crate::{
    device::{Cpu, Device},
    kind::Kind,
    shapes::shape::Shape,
    utils::Sealed,
};

pub trait ToTchTensor {
    fn to_tch(&self) -> &tch::Tensor;
}

impl<S: Shape, D: Device, K: Kind, G: RequiresGrad> ToTchTensor for Tensor<S, D, K, G> {
    fn to_tch(&self) -> &tch::Tensor {
        &self.repr
    }
}

pub trait RequiresGrad: Sealed {
    const REQUIRES_GRAD: bool;
}
pub struct Grad;
pub struct NoGrad;
impl Sealed for Grad {}
impl Sealed for NoGrad {}
impl RequiresGrad for Grad {
    const REQUIRES_GRAD: bool = true;
}
impl RequiresGrad for NoGrad {
    const REQUIRES_GRAD: bool = false;
}

pub struct Initialized;
pub struct Uninitialized;
impl Sealed for Initialized {}
impl Sealed for Uninitialized {}

#[must_use]
pub struct Tensor<
    S: Shape,
    D: Device = Cpu,
    K: Kind = f32,
    G: RequiresGrad = NoGrad,
    I: Sealed = Initialized,
> {
    pub(crate) repr: tch::Tensor,
    pub(crate) shape: PhantomData<S>,
    pub(crate) device: PhantomData<D>,
    pub(crate) dtype: PhantomData<K>,
    pub(crate) grad: PhantomData<G>,
    pub(crate) init: PhantomData<I>,
}

impl<S: Shape, D: Device, K: Kind, G: RequiresGrad, I: Sealed> Tensor<S, D, K, G, I> {
    pub const fn dims(&self) -> i64 {
        S::DIMS
    }

    pub const fn nelems(&self) -> usize {
        S::NELEMS
    }

    pub const fn requires_grad(&self) -> bool {
        G::REQUIRES_GRAD
    }

    pub fn shape(&self) -> &[i64] {
        S::dims()
    }

    pub fn to_tch_mut(&mut self) -> &mut tch::Tensor {
        &mut self.repr
    }

    pub fn print(&self) {
        self.repr.print();
    }
}

impl<S: Shape, D: Device, K: Kind> Tensor<S, D, K, NoGrad> {
    pub fn set_require_grad(&self) -> Tensor<S, D, K, Grad> {
        Tensor {
            repr: self.repr.set_requires_grad(true),
            ..Default::default()
        }
    }
}

impl<S: Shape, D: Device, K: Kind> Tensor<S, D, K, Grad> {
    pub fn set_not_require_grad(&self) -> Tensor<S, D, K, NoGrad> {
        Tensor {
            repr: self.repr.set_requires_grad(false),
            ..Default::default()
        }
    }
}

impl<S: Shape, D: Device, K: Kind, G: RequiresGrad> Default for Tensor<S, D, K, G> {
    fn default() -> Self {
        let mut repr = tch::Tensor::empty(S::dims(), (K::into_dtype(), D::into_device()));
        let _ = repr.requires_grad_(G::REQUIRES_GRAD);
        Self {
            repr,
            shape: PhantomData,
            device: PhantomData,
            dtype: PhantomData,
            grad: PhantomData,
            init: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {}
