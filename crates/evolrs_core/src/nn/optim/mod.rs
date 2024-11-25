use tch::nn::OptimizerConfig;

use crate::{device::Device, shapes::shape::Shape, tensor::Tensor};

use super::vs::Vs;

pub trait Backward<T> {
    const BACKWARD_CHECK: ();
    fn backward_step(&mut self, loss: &T);
}

pub struct Sgd {
    repr: tch::nn::Optimizer,
}

impl Sgd {
    pub fn new(vs: &Vs, lr: f64) -> Result<Self, tch::TchError> {
        Ok(Self {
            repr: tch::nn::Sgd::default().build(vs.vs(), lr)?,
        })
    }
}

impl<S: Shape, D: Device> Backward<Tensor<S, D, f32>> for Sgd {
    const BACKWARD_CHECK: () = assert!(
        S::NELEMS == 1,
        "The loss must be a Scalar or a Tensor with only 1 element"
    );

    fn backward_step(&mut self, loss: &Tensor<S, D, f32>) {
        #![allow(path_statements)]
        <Sgd as Backward<Tensor<S, D, f32>>>::BACKWARD_CHECK;
        self.repr.backward_step(&loss.repr);
    }
}
