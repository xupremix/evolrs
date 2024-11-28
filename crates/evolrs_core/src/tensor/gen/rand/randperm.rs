use crate::{
    device::Device,
    kind::Kind,
    shapes::shape::Rank1,
    tensor::{NoGrad, Tensor},
};

impl<const PERMS: usize, D: Device, K: Kind> Tensor<Rank1<PERMS>, D, K, NoGrad> {
    pub fn randperm() -> Self {
        Self {
            repr: tch::Tensor::randperm(PERMS as i64, (K::into_dtype(), D::into_device())),
            ..Default::default()
        }
    }
}
