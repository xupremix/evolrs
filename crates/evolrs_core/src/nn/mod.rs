use crate::{device::Device, shapes::shape::Shape, tensor::Tensor};

pub mod modules;
pub mod optim;
pub mod vs;

pub trait Forward<const I: usize, const O: usize>: Shape {
    type ForwardShape: Shape;
}

pub trait Module<const I: usize, const O: usize, D: Device> {
    fn forward<S: Forward<I, O>>(&self, xs: &Tensor<S, D, f32>) -> Tensor<S::ForwardShape, D, f32>;
}

#[cfg(test)]
mod tests {}
