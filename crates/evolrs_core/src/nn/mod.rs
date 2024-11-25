use crate::{
    device::Device,
    shapes::shape::Shape,
    tensor::{Tensor, ToTchTensor},
};

pub mod modules;
pub mod optim;
pub mod vs;

pub trait Forward<const I: usize, const O: usize>: Shape {
    type ForwardShape: Shape;
}

pub trait Module<S: Shape, D: Device> {
    type Output: ToTchTensor;
    fn forward(&self, xs: &Tensor<S, D, f32>) -> Self::Output;
}

#[cfg(test)]
mod tests {}
