use crate::device::Device;
use crate::kind::Kind;
use crate::shapes::shape::Shape;
use crate::tensor::Tensor;

impl<S: Shape, D: Device, K: Kind> Clone for Tensor<S, D, K> {
    fn clone(&self) -> Self {
        Self {
            repr: self.repr.copy(),
            ..Default::default()
        }
    }
}
