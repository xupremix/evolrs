use std::marker::PhantomData;

use tch::nn::{OptimizerConfig, VarStore};

use crate::{device::Device, shapes::shape::Shape, tensor::Tensor};

pub trait Backward<T> {
    const CHECK: ();
    fn backward_step(&mut self, loss: &T);
}

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
}

impl<S: Shape, D: Device> Backward<Tensor<S, D, f32>> for Sgd<D> {
    const CHECK: () = assert!(
        S::NELEMS == 1,
        "The loss must be a Scalar or a Tensor with only 1 element"
    );

    fn backward_step(&mut self, loss: &Tensor<S, D, f32>) {
        self.repr.backward_step(&loss.repr);
    }
}
