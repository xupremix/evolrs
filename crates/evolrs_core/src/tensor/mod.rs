use std::marker::PhantomData;

pub mod gen;
pub mod wrap;

use crate::{
    device::{Cpu, Device},
    kind::Kind,
    shapes::shape::Shape,
};

#[must_use]
pub struct Tensor<S: Shape, D: Device = Cpu, K: Kind = f32> {
    pub(crate) repr: tch::Tensor,
    pub(crate) shape: PhantomData<S>,
    pub(crate) device: PhantomData<D>,
    pub(crate) dtype: PhantomData<K>,
}

impl<S: Shape, D: Device, K: Kind> Tensor<S, D, K> {
    pub fn dims(&self) -> i64 {
        S::DIMS
    }

    pub fn nelems(&self) -> usize {
        S::NELEMS
    }

    pub fn to_tch(&self) -> &tch::Tensor {
        &self.repr
    }

    pub fn to_tch_mut(&mut self) -> &mut tch::Tensor {
        &mut self.repr
    }
}

impl<S: Shape, D: Device, K: Kind> Default for Tensor<S, D, K> {
    fn default() -> Self {
        Self {
            repr: tch::Tensor::default(),
            shape: PhantomData,
            device: PhantomData,
            dtype: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {}
