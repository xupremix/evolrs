use crate::{
    device::Device,
    kind::Kind,
    shapes::shape::{Rank1, Scalar},
    tensor::Tensor,
};

// TODO: check if logspace inherits gradient tracking or not

pub trait LogspaceScalar {
    fn logspace<S: Into<tch::Scalar> + PartialOrd>(start: S, end: S, base: f64) -> Self;
}

pub trait LogspaceTensor<T> {
    fn logspace(start: T, end: T, base: f64) -> Self;
}

pub trait LogspaceScalarTensor<T> {
    fn logspace<S: Into<tch::Scalar> + PartialOrd>(start: S, end: T, base: f64) -> Self;
}

pub trait LogspaceTensorScalar<T> {
    fn logspace<S: Into<tch::Scalar> + PartialOrd>(start: T, end: S, base: f64) -> Self;
}

impl<const STEPS: usize, D: Device, K: Kind> LogspaceScalar for Tensor<Rank1<STEPS>, D, K> {
    fn logspace<S: Into<tch::Scalar> + PartialOrd>(start: S, end: S, base: f64) -> Self {
        Self {
            repr: tch::Tensor::logspace(
                start,
                end,
                STEPS as i64,
                base,
                (K::into_dtype(), D::into_device()),
            ),
            ..Default::default()
        }
    }
}

impl<const STEPS: usize, D: Device, K: Kind> LogspaceTensor<Tensor<Scalar, D, K>>
    for Tensor<Rank1<STEPS>, D, K>
{
    fn logspace(start: Tensor<Scalar, D, K>, end: Tensor<Scalar, D, K>, base: f64) -> Self {
        Self {
            repr: tch::Tensor::logspace_tensor_tensor(
                &start.repr,
                &end.repr,
                STEPS as i64,
                base,
                (K::into_dtype(), D::into_device()),
            ),
            ..Default::default()
        }
    }
}

impl<const STEPS: usize, D: Device, K: Kind> LogspaceScalarTensor<Tensor<Scalar, D, K>>
    for Tensor<Rank1<STEPS>, D, K>
{
    fn logspace<S: Into<tch::Scalar> + PartialOrd>(
        start: S,
        end: Tensor<Scalar, D, K>,
        base: f64,
    ) -> Self {
        Self {
            repr: tch::Tensor::logspace_scalar_tensor(
                start,
                &end.repr,
                STEPS as i64,
                base,
                (K::into_dtype(), D::into_device()),
            ),
            ..Default::default()
        }
    }
}

impl<const STEPS: usize, D: Device, K: Kind> LogspaceTensorScalar<Tensor<Scalar, D, K>>
    for Tensor<Rank1<STEPS>, D, K>
{
    fn logspace<S: Into<tch::Scalar> + PartialOrd>(
        start: Tensor<Scalar, D, K>,
        end: S,
        base: f64,
    ) -> Self {
        Self {
            repr: tch::Tensor::logspace_tensor_scalar(
                &start.repr,
                end,
                STEPS as i64,
                base,
                (K::into_dtype(), D::into_device()),
            ),
            ..Default::default()
        }
    }
}
