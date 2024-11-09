use crate::{device::Device, kind::Kind, shapes::shape::Rank1, tensor::Tensor};

pub trait Arange<T> {
    fn arange() -> Self;
}

impl<const END: usize, D: Device, K: Kind> Arange<f64> for Tensor<Rank1<END>, D, K> {
    fn arange() -> Self {
        Self {
            repr: tch::Tensor::arange(END as f64, (K::into_dtype(), D::into_device())),
            ..Default::default()
        }
    }
}
impl<const END: usize, D: Device, K: Kind> Arange<i64> for Tensor<Rank1<END>, D, K> {
    fn arange() -> Self {
        Self {
            repr: tch::Tensor::arange(END as i64, (K::into_dtype(), D::into_device())),
            ..Default::default()
        }
    }
}
