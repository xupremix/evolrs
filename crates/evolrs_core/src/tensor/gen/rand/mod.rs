use crate::{
    device::Device,
    kind::{Kind, Rand},
    shapes::shape::Shape,
    tensor::Tensor,
};

pub mod randint;
pub mod randn;
pub mod randperm;

impl<S: Shape, D: Device, K: Kind + Rand> Tensor<S, D, K> {
    pub fn rand() -> Self {
        Self {
            repr: tch::Tensor::rand(S::dims(), (K::into_dtype(), D::into_device())),
            ..Default::default()
        }
    }

    pub fn rand_like(&self) -> Self {
        Self {
            repr: self.repr.rand_like(),
            ..Default::default()
        }
    }
}
