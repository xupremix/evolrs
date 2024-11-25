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

pub trait Module<T> {
    type Output: ToTchTensor;
    fn forward(&self, xs: &T) -> Self::Output;
}

#[cfg(test)]
mod tests {}
