use std::marker::PhantomData;

use crate::{
    device::Device,
    shapes::shape::Shape,
    tensor::{FromTchTensor, Tensor, ToTchTensor},
};

pub mod build;
pub mod modules;
pub mod optim;
pub mod vs;

use build::ModelBuilder;
use tch::nn::{Module as _, Sequential};
use vs::Vs;

pub trait Forward<const I: usize, const O: usize>: Shape {
    type ForwardShape: Shape;
}

pub trait Module<T> {
    type Output: FromTchTensor;
    fn forward(&self, xs: &T) -> Self::Output;
}

pub struct Model<M> {
    repr: Sequential,
    module: PhantomData<M>,
}

impl<M> Model<M> {
    pub fn print(&self) {
        println!("{:#?}", self.repr);
    }
}

impl<T, M: Module<T>> Module<T> for Model<M>
where
    T: ToTchTensor,
{
    type Output = M::Output;

    fn forward(&self, xs: &T) -> Self::Output {
        M::Output::from_tch_tensor(self.repr.forward(xs.to_tch_tensor()))
    }
}

// Testing implementation of Module for a tuple of modules

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
}

// Testing implementation of Model Builder for a tuple of Model Builders

impl<M0: ModelBuilder, M1: ModelBuilder, M2: ModelBuilder> ModelBuilder for (M0, M1, M2) {
    type Config = (M0::Config, M1::Config, M2::Config);

    fn step(vs: &Vs, c: Self::Config, seq: Sequential) -> Sequential {
        let (c0, c1, c2) = c;
        let seq = M0::step(vs, c0, seq);
        let seq = M1::step(vs, c1, seq);
        M2::step(vs, c2, seq)
    }
}

#[cfg(test)]
mod tests {}
