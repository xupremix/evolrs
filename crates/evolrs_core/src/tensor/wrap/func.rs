use crate::{
    device::Device,
    kind::{scalar::IntoScalar, Kind},
    shapes::shape::Shape,
    tensor::{Initialized, RequiresGrad, Tensor},
};

impl<S: Shape, D: Device, K: Kind, G: RequiresGrad> Tensor<S, D, K, G, Initialized> {
    pub fn gelu(&mut self) -> Tensor<S, D, K, G> {
        Tensor {
            repr: self.repr.gelu(""),
            ..Default::default()
        }
    }

    pub fn gelu_(&mut self) -> Tensor<S, D, K, G> {
        Tensor {
            repr: self.repr.gelu_(""),
            ..Default::default()
        }
    }

    pub fn gelu_tanh(&mut self) -> Tensor<S, D, K, G> {
        Tensor {
            repr: self.repr.gelu("tanh"),
            ..Default::default()
        }
    }

    pub fn gelu_tanh_(&mut self) -> Tensor<S, D, K, G> {
        Tensor {
            repr: self.repr.gelu_("tanh"),
            ..Default::default()
        }
    }

    pub fn threashold<V: IntoScalar>(&self, threshold: V, value: V) -> Tensor<S, D, K> {
        Tensor {
            repr: self.repr.threshold(threshold.into(), value.into()),
            ..Default::default()
        }
    }
    pub fn threashold_<V: IntoScalar>(&mut self, threshold: V, value: V) -> Tensor<S, D, K> {
        Tensor {
            repr: self.repr.threshold_(threshold.into(), value.into()),
            ..Default::default()
        }
    }

    // TODO: check why there isn't an inplace version of this fn
    pub fn log_sigmoid(&self) -> Tensor<S, D, K> {
        Tensor {
            repr: self.repr.log_sigmoid(),
            ..Default::default()
        }
    }
    pub fn softplus(&self) -> Tensor<S, D, K> {
        Tensor {
            repr: self.repr.softplus(),
            ..Default::default()
        }
    }
    pub fn softshrink(&self) -> Tensor<S, D, K> {
        Tensor {
            repr: self.repr.softshrink(),
            ..Default::default()
        }
    }
    pub fn hardshrink(&self) -> Tensor<S, D, K> {
        Tensor {
            repr: self.repr.hardshrink(),
            ..Default::default()
        }
    }
}
