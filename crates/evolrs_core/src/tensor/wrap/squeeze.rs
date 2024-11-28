use crate::{
    device::Device,
    kind::Kind,
    shapes::shape::Shape,
    tensor::{RequiresGrad, Tensor},
};

pub trait Squeeze<S: Shape>: Shape {
    const CHECK: ();
}

pub trait SqueezeDim<const DIM: usize>: Shape {
    type SqueezeShape: Shape;
}

impl<S: Shape, D: Device, K: Kind, G: RequiresGrad> Tensor<S, D, K, G> {
    pub fn squeeze<Dst: Squeeze<S>>(&self) -> Tensor<Dst, D, K, G> {
        #![allow(path_statements)]
        Dst::CHECK;
        Tensor {
            repr: self.repr.squeeze(),
            ..Default::default()
        }
    }

    pub fn squeeze_dim<const DIM: usize>(&self) -> Tensor<S::SqueezeShape, D, K, G>
    where
        S: SqueezeDim<DIM>,
    {
        Tensor {
            repr: self.repr.squeeze_dim(DIM as i64),
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {}
