use crate::{device::Device, kind::Kind, shapes::shape::Shape, tensor::Tensor};

pub mod item;
pub mod math;

impl<S: Shape, D: Device, K: Kind> Tensor<S, D, K> {
    pub fn print(&self) {
        self.repr.print();
    }
}
