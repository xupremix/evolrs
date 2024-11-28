use crate::{
    device::Device,
    kind::Kind,
    shapes::shape::Shape,
    tensor::{NoGrad, Tensor},
};

pub trait Full<T> {
    fn full(value: T) -> Self;
}

impl<S: Shape, D: Device, K: Kind, T: Into<tch::Scalar>> Full<T> for Tensor<S, D, K, NoGrad> {
    fn full(value: T) -> Self {
        Self {
            repr: tch::Tensor::full(S::dims(), value, (K::into_dtype(), D::into_device())),
            ..Default::default()
        }
    }
}
