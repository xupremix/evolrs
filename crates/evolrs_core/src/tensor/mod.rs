use std::marker::PhantomData;

use crate::{device::Device, dtype::Kind, shapes::shape::Shape};

pub struct Tensor<S: Shape, D: Device, K: Kind> {
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
}

impl<S: Shape, D: Device, K: Kind> Default for Tensor<S, D, K> {
    fn default() -> Self {
        Self::new()
    }
}
