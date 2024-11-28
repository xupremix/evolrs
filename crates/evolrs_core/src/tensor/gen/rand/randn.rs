use crate::{
    device::Device,
    kind::restriction::composite::FloatOrComplex,
    shapes::shape::Shape,
    tensor::{NoGrad, RequiresGrad, Tensor},
};

impl<S: Shape, D: Device, K: FloatOrComplex> Tensor<S, D, K, NoGrad> {
    pub fn randn() -> Self {
        Self {
            repr: tch::Tensor::randn(S::dims(), (K::into_dtype(), D::into_device())),
            ..Default::default()
        }
    }
}
impl<S: Shape, D: Device, K: FloatOrComplex, G: RequiresGrad> Tensor<S, D, K, G> {
    pub fn randn_like(&self) -> Tensor<S, D, K, NoGrad> {
        Tensor {
            repr: self.repr.randn_like(),
            ..Default::default()
        }
    }
}
