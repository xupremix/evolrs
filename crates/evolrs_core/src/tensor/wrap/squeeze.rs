use crate::{device::Device, kind::Kind, shapes::shape::Shape, tensor::Tensor};

pub trait Squeeze<S: Shape>: Shape {
    const CHECK: ();
}

impl<S: Shape, D: Device, K: Kind> Tensor<S, D, K> {
    pub fn squeeze<Dst: Squeeze<S>>(&self) -> Tensor<Dst, D, K> {
        #![allow(path_statements)]
        Dst::CHECK;
        Tensor {
            repr: self.repr.squeeze(),
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {}
