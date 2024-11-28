use crate::{
    device::Device,
    kind::restriction::composite::FloatOrComplex,
    shapes::shape::Shape,
    tensor::{NoGrad, RequiresGrad, Tensor},
};

pub mod randint;
pub mod randn;
pub mod randperm;

impl<S: Shape, D: Device, K: FloatOrComplex> Tensor<S, D, K, NoGrad> {
    pub fn rand() -> Self {
        Self {
            repr: tch::Tensor::rand(S::dims(), (K::into_dtype(), D::into_device())),
            ..Default::default()
        }
    }
}
impl<S: Shape, D: Device, K: FloatOrComplex, G: RequiresGrad> Tensor<S, D, K, G> {
    pub fn rand_like(&self) -> Tensor<S, D, K, NoGrad> {
        Tensor {
            repr: self.repr.rand_like(),
            ..Default::default()
        }
    }
}
