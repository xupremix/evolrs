use std::marker::PhantomData;

use crate::{
    device::Device,
    kind::Kind,
    shapes::shape::Shape,
    tensor::{RequiresGrad, Tensor, ToTchTensor},
};

pub mod modules;
pub mod optim;
pub mod vs;

use tch::nn::{seq, Module as _, Sequential};
use vs::Vs;

pub trait Forward<const I: usize, const O: usize>: Shape {
    type ForwardShape: Shape;
}

pub(crate) trait FromTchTensor {
    fn from_tch_tensor(repr: tch::Tensor) -> Self;
}

impl<S: Shape, D: Device, K: Kind, G: RequiresGrad> FromTchTensor for Tensor<S, D, K, G> {
    fn from_tch_tensor(repr: tch::Tensor) -> Self {
        Tensor {
            repr,
            ..Default::default()
        }
    }
}

pub trait Module<T>: Sized {
    type Output: FromTchTensor;
    fn forward(&self, xs: &T) -> Self::Output;

    type Config;
    fn build(vs: &Vs, c: Self::Config) -> Model<Self> {
        let repr = seq();
        let repr = Self::step(vs, c, repr);
        Model {
            repr,
            module: PhantomData,
        }
    }
    fn step(vs: &Vs, c: Self::Config, seq: Sequential) -> Sequential;
}

pub struct Model<M> {
    repr: Sequential,
    module: PhantomData<M>,
}

impl<T, M: Module<T>> Module<T> for Model<M>
where
    T: ToTchTensor,
{
    type Output = M::Output;

    fn forward(&self, xs: &T) -> Self::Output {
        M::Output::from_tch_tensor(self.repr.forward(xs.to_tch_tensor()))
    }

    type Config = M::Config;

    fn step(vs: &Vs, c: Self::Config, seq: Sequential) -> Sequential {
        M::step(vs, c, seq)
    }
}

impl<S, D, M0, M1, M2> Module<Tensor<S, D, f32>> for (M0, M1, M2)
where
    S: Shape,
    D: Device,
    M0: Module<Tensor<S, D, f32>>,
    M1: Module<M0::Output>,
    M2: Module<M1::Output>,
{
    type Output = M2::Output;

    fn forward(&self, xs: &Tensor<S, D, f32>) -> Self::Output {
        let xs = self.0.forward(xs);
        let xs = self.1.forward(&xs);
        self.2.forward(&xs)
    }

    type Config = (M0::Config, M1::Config, M2::Config);
    fn step(vs: &Vs, c: Self::Config, seq: tch::nn::Sequential) -> tch::nn::Sequential {
        let seq = M0::step(vs, c.0, seq);
        let seq = M1::step(vs, c.1, seq);
        M2::step(vs, c.2, seq)
    }
}

#[cfg(test)]
mod tests {}
