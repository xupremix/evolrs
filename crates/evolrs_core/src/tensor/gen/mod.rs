use crate::device::Device;
use crate::kind::Kind;
use crate::tensor::Shape;
use crate::tensor::Tensor;

pub mod arange;
pub mod eye;
pub mod full;
pub mod linspace;
pub mod logspace;

impl<S: Shape, D: Device, K: Kind> Tensor<S, D, K> {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn new_like(&self) -> Self {
        Default::default()
    }

    pub fn zeros() -> Self {
        Self {
            repr: tch::Tensor::zeros(S::dims(), (K::into_dtype(), D::into_device())),
            ..Default::default()
        }
    }

    pub fn zeros_like(&self) -> Self {
        Self {
            repr: tch::Tensor::zeros_like(&self.repr),
            ..Default::default()
        }
    }

    pub fn ones() -> Self {
        Self {
            repr: tch::Tensor::ones(S::dims(), (K::into_dtype(), D::into_device())),
            ..Default::default()
        }
    }

    pub fn ones_like(&self) -> Self {
        Self {
            repr: tch::Tensor::ones_like(&self.repr),
            ..Default::default()
        }
    }
}
