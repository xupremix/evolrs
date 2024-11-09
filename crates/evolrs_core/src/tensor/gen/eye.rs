use crate::{device::Device, kind::Kind, shapes::shape::Rank2, tensor::Tensor};

impl<const D0: usize, const D1: usize, D: Device, K: Kind> Tensor<Rank2<D0, D1>, D, K> {
    pub fn eye() -> Self {
        Self {
            repr: tch::Tensor::eye_m(D0 as i64, D1 as i64, (K::into_dtype(), D::into_device())),
            ..Default::default()
        }
    }
}
