use crate::{device::Device, kind::Kind, shapes::shape::Shape, tensor::Tensor};

impl<S: Shape, D: Device, K: Kind, K2: Kind> PartialEq<Tensor<S, D, K2>> for Tensor<S, D, K> {
    fn eq(&self, other: &Tensor<S, D, K2>) -> bool {
        self.repr == other.repr
    }
}
