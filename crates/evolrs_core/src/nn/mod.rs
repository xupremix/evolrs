use crate::{
    device::{Cpu, Device},
    kind::Kind,
    shapes::shape::{Rank2, Shape},
    tensor::{wrap::matmul::Matmul, Tensor},
};

pub mod modules;
pub mod optim;

pub trait Module<const I: usize, const O: usize, D: Device = Cpu, K: Kind = f32> {
    fn forward<S: Shape>(
        &self,
        xs: Tensor<S, D, K>,
    ) -> Tensor<<Rank2<I, O> as Matmul<S>>::MatmulShape, D, K>
    where
        Rank2<I, O>: Matmul<S>;
}
