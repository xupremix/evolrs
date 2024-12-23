use std::marker::PhantomData;

use crate::device::Device;
use crate::kind::Kind;
use crate::tensor::Shape;
use crate::tensor::Tensor;

use super::{NoGrad, RequiresGrad, Sealed, Uninitialized};

pub mod arange;
pub mod dist;
pub mod eye;
pub mod full;
pub mod linspace;
pub mod logspace;
pub mod rand;

impl<S: Shape, D: Device, K: Kind, G: RequiresGrad, I: Sealed> Tensor<S, D, K, G, I> {
    pub fn new_like(&self) -> Tensor<S, D, K, NoGrad, Uninitialized> {
        Tensor::new()
    }

    pub fn empty_like(&self) -> Tensor<S, D, K, NoGrad> {
        Tensor {
            repr: tch::Tensor::empty_like(&self.repr),
            ..Default::default()
        }
    }

    pub fn zeros_like(&self) -> Tensor<S, D, K, NoGrad> {
        Tensor {
            repr: tch::Tensor::zeros_like(&self.repr),
            ..Default::default()
        }
    }

    pub fn ones_like(&self) -> Tensor<S, D, K, NoGrad> {
        Tensor {
            repr: tch::Tensor::ones_like(&self.repr),
            ..Default::default()
        }
    }
}

impl<S: Shape, D: Device, K: Kind> Tensor<S, D, K, NoGrad, Uninitialized> {
    pub fn new() -> Self {
        Self {
            repr: tch::Tensor::new(),
            shape: PhantomData,
            device: PhantomData,
            dtype: PhantomData,
            grad: PhantomData,
            init: PhantomData,
        }
    }
}

impl<S: Shape, D: Device, K: Kind> Tensor<S, D, K, NoGrad> {
    pub fn empty() -> Self {
        Self {
            repr: tch::Tensor::empty(S::dims(), (K::into_dtype(), D::into_device())),
            ..Default::default()
        }
    }

    pub fn zeros() -> Self {
        Self {
            repr: tch::Tensor::zeros(S::dims(), (K::into_dtype(), D::into_device())),
            ..Default::default()
        }
    }

    pub fn ones() -> Self {
        Self {
            repr: tch::Tensor::ones(S::dims(), (K::into_dtype(), D::into_device())),
            ..Default::default()
        }
    }
}
