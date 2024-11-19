use crate::{
    device::Device, kind::restriction::composite::FloatOrComplex, shapes::shape::Shape,
    tensor::Tensor,
};

impl<S: Shape, D: Device, K: FloatOrComplex> Tensor<S, D, K> {
    pub fn randn() -> Self {
        Self {
            repr: tch::Tensor::randn(S::dims(), (K::into_dtype(), D::into_device())),
            ..Default::default()
        }
    }

    pub fn randn_like(&self) -> Self {
        Self {
            repr: self.repr.randn_like(),
            ..Default::default()
        }
    }
}
