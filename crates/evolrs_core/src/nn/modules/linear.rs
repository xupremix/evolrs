use std::{borrow::Borrow, marker::PhantomData};

use tch::nn::{LinearConfig, Module as _};

use crate::{
    device::{Cpu, Device},
    nn::{Forward, Module},
    tensor::Tensor,
};

#[derive(Debug)]
pub struct Linear<const I: usize, const O: usize, D: Device = Cpu> {
    repr: tch::nn::Linear,
    device: PhantomData<D>,
}

impl<const I: usize, const O: usize, D: Device> Linear<I, O, D> {
    pub fn new<'a, V: Borrow<tch::nn::Path<'a>>>(vs: V, config: LinearConfig) -> Self {
        Self {
            repr: tch::nn::linear(vs, I as i64, O as i64, config),
            device: PhantomData,
        }
    }
}

impl<const I: usize, const O: usize, S: Forward<I, O>, D: Device> Module<S, D> for Linear<I, O, D> {
    type Output = Tensor<S::ForwardShape, D, f32>;

    fn forward(&self, xs: &Tensor<S, D, f32>) -> Self::Output {
        Tensor {
            repr: self.repr.forward(&xs.repr),
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {}
