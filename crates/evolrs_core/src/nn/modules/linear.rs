use std::{borrow::Borrow, marker::PhantomData};

use tch::nn::{LinearConfig, Module as _};

use crate::{
    device::{Cpu, Device},
    kind::Kind,
    nn::Module,
    ops::method_traits::matmul::Matmul,
    shapes::shape::{Rank2, Shape},
    tensor::Tensor,
};

#[derive(Debug)]
pub struct Linear<const I: usize, const O: usize, D: Device = Cpu, K: Kind = f32> {
    repr: tch::nn::Linear,
    device: PhantomData<D>,
    kind: PhantomData<K>,
}

impl<const I: usize, const O: usize, D: Device, K: Kind> Linear<I, O, D, K> {
    pub fn new<'a, V: Borrow<tch::nn::Path<'a>>>(vs: V, config: LinearConfig) -> Self {
        Self {
            repr: tch::nn::linear(vs, I as i64, O as i64, config),
            device: PhantomData,
            kind: PhantomData,
        }
    }
}

impl<const I: usize, const O: usize, D: Device, K: Kind> Module<I, O, D, K> for Linear<I, O, D, K> {
    fn forward<S: Shape>(
        &self,
        xs: Tensor<S, D, K>,
    ) -> Tensor<<Rank2<I, O> as Matmul<S>>::MatmulShape, D, K>
    where
        Rank2<I, O>: Matmul<S>,
    {
        Tensor {
            repr: self.repr.forward(&xs.repr),
            ..Default::default()
        }
    }
}

// TODO: Add tests after checkign the implementation of module and vs
#[cfg(test)]
mod tests {}
