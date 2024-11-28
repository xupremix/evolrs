use crate::{
    device::Device,
    kind::Kind,
    shapes::shape::{Rank1, Scalar},
    tensor::Tensor,
};

// TODO: check if linspace inherits gradient tracking or not

pub trait LinspaceScalar {
    fn linspace<S: Into<tch::Scalar> + PartialOrd>(start: S, end: S) -> Self;
}

pub trait LinspaceTensor<T> {
    fn linspace(start: T, end: T) -> Self;
}

pub trait LinspaceScalarTensor<T> {
    fn linspace<S: Into<tch::Scalar> + PartialOrd>(start: S, end: T) -> Self;
}

pub trait LinspaceTensorScalar<T> {
    fn linspace<S: Into<tch::Scalar> + PartialOrd>(start: T, end: S) -> Self;
}

impl<const STEPS: usize, D: Device, K: Kind> LinspaceScalar for Tensor<Rank1<STEPS>, D, K> {
    fn linspace<S: Into<tch::Scalar> + PartialOrd>(start: S, end: S) -> Self {
        assert!(start <= end, "start must be less than or equal to end");
        Self {
            repr: tch::Tensor::linspace(
                start,
                end,
                STEPS as i64,
                (K::into_dtype(), D::into_device()),
            ),
            ..Default::default()
        }
    }
}

impl<const STEPS: usize, D: Device, K: Kind> LinspaceTensor<Tensor<Scalar, D, K>>
    for Tensor<Rank1<STEPS>, D, K>
{
    fn linspace(start: Tensor<Scalar, D, K>, end: Tensor<Scalar, D, K>) -> Self {
        Self {
            repr: tch::Tensor::linspace_tensor_tensor(
                &start.repr,
                &end.repr,
                STEPS as i64,
                (K::into_dtype(), D::into_device()),
            ),
            ..Default::default()
        }
    }
}

impl<const STEPS: usize, D: Device, K: Kind> LinspaceScalarTensor<Tensor<Scalar, D, K>>
    for Tensor<Rank1<STEPS>, D, K>
{
    fn linspace<S: Into<tch::Scalar> + PartialOrd>(start: S, end: Tensor<Scalar, D, K>) -> Self {
        Self {
            repr: tch::Tensor::linspace_scalar_tensor(
                start,
                &end.repr,
                STEPS as i64,
                (K::into_dtype(), D::into_device()),
            ),
            ..Default::default()
        }
    }
}

impl<const STEPS: usize, D: Device, K: Kind> LinspaceTensorScalar<Tensor<Scalar, D, K>>
    for Tensor<Rank1<STEPS>, D, K>
{
    fn linspace<S: Into<tch::Scalar> + PartialOrd>(start: Tensor<Scalar, D, K>, end: S) -> Self {
        Self {
            repr: tch::Tensor::linspace_tensor_scalar(
                &start.repr,
                end,
                STEPS as i64,
                (K::into_dtype(), D::into_device()),
            ),
            ..Default::default()
        }
    }
}
