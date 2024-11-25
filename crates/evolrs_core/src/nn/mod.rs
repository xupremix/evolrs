use crate::shapes::shape::Shape;

pub mod modules;
pub mod optim;
pub mod vs;

pub trait Forward<const I: usize, const O: usize>: Shape {
    type ForwardShape: Shape;
}

pub trait Module<T> {
    type Output;
    fn forward(&self, xs: &T) -> Self::Output;
}

#[cfg(test)]
mod tests {}
