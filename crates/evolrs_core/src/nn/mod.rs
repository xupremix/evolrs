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

use tch::nn::{Module as _, Sequential};

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

impl<T, M: Module<T>> Module<T> for Model<M>
where
    T: ToTchTensor,
{
    type Output = M::Output;

    fn forward(&self, xs: &T) -> Self::Output {
        M::Output::from_tch_tensor(self.repr.forward(xs.to_tch_tensor()))
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
}

#[cfg(test)]
mod tests {}
