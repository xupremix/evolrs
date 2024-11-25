use std::marker::PhantomData;

use tch::nn::{OptimizerConfig, VarStore};

use crate::{device::Device, shapes::shape::Shape, tensor::ToTchTensor as _};

use super::Module;

pub struct Sgd<D: Device> {
    repr: tch::nn::Optimizer,
    device: PhantomData<D>,
}

impl<D: Device> Sgd<D> {
    pub fn new(vs: &VarStore, lr: f64) -> Result<Self, tch::TchError> {
        Ok(Self {
            repr: tch::nn::Sgd::default().build(vs, lr)?,
            device: PhantomData,
        })
    }

    pub fn backward<S: Shape, M: Module<S, D>>(&mut self, loss: &M::Output) {
        self.repr.backward_step(loss.to_tch());
    }
}
