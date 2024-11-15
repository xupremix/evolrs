use crate::{device::Device, kind::Kind, shapes::shape::Rank1, tensor::Tensor};

impl<const PERMS: usize, D: Device, K: Kind> Tensor<Rank1<PERMS>, D, K> {
    pub fn randperm() -> Self {
        Self {
            repr: tch::Tensor::randperm(PERMS as i64, (K::into_dtype(), D::into_device())),
            ..Default::default()
        }
    }
}
