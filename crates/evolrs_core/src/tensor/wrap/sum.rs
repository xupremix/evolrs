use crate::{
    device::Device,
    kind::Kind,
    shapes::shape::{Scalar, Shape},
    tensor::Tensor,
};

pub trait SumShape<const DIM: i64, const KEEP_DIM: bool>: Shape {
    type SumShape: Shape;
}

impl<S: Shape, D: Device, K: Kind> Tensor<S, D, K> {
    pub fn sum(&self) -> Tensor<Scalar, D, K> {
        Tensor {
            repr: self.repr.sum(None),
            ..Default::default()
        }
    }

    pub fn sum_dim<const DIM: i64, const KEEP_DIM: bool>(&self) -> Tensor<S::SumShape, D, K>
    where
        S: SumShape<DIM, KEEP_DIM>,
    {
        Tensor {
            repr: self.repr.sum_dim_intlist(DIM, KEEP_DIM, None),
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {}
