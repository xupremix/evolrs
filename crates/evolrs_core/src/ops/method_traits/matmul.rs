use crate::{device::Device, kind::Kind, shapes::shape::Shape, tensor::Tensor};

pub trait Matmul<Rhs: Shape>: Shape {
    type MatmulShape: Shape;
}

impl<S: Shape, D: Device, K: Kind> Tensor<S, D, K> {
    pub fn matmul<Rhs: Matmul<S>>(
        &self,
        rhs: &Tensor<Rhs, D, K>,
    ) -> Tensor<Rhs::MatmulShape, D, K> {
        Tensor {
            repr: self.repr.matmul(&rhs.repr),
            ..Default::default()
        }
    }
}
