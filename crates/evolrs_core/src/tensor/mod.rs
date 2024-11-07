use std::marker::PhantomData;

use crate::{
    device::{Cpu, Device},
    dtype::Kind,
    shapes::shape::Shape,
};

pub struct Tensor<S: Shape, D: Device = Cpu, K: Kind = f32> {
    pub(crate) repr: tch::Tensor,
    pub(crate) shape: PhantomData<S>,
    pub(crate) device: PhantomData<D>,
    pub(crate) dtype: PhantomData<K>,
}

impl<S: Shape, D: Device, K: Kind> Tensor<S, D, K> {
    pub fn new() -> Self {
        Self {
            repr: tch::Tensor::new(),
            shape: PhantomData,
            device: PhantomData,
            dtype: PhantomData,
        }
    }

    pub fn zeros() -> Self {
        Self {
            repr: tch::Tensor::zeros(S::dims(), (K::into_dtype(), D::into_device())),
            shape: PhantomData,
            device: PhantomData,
            dtype: PhantomData,
        }
    }

    pub fn zeros_like(&self) -> Self {
        Self {
            repr: tch::Tensor::zeros_like(&self.repr),
            shape: PhantomData,
            device: PhantomData,
            dtype: PhantomData,
        }
    }

    pub fn ones() -> Self {
        Self {
            repr: tch::Tensor::ones(S::dims(), (K::into_dtype(), D::into_device())),
            shape: PhantomData,
            device: PhantomData,
            dtype: PhantomData,
        }
    }

    pub fn ones_like(&self) -> Self {
        Self {
            repr: tch::Tensor::ones_like(&self.repr),
            shape: PhantomData,
            device: PhantomData,
            dtype: PhantomData,
        }
    }
}

impl<S: Shape, D: Device, K: Kind> Default for Tensor<S, D, K> {
    fn default() -> Self {
        Self::new()
    }
}
