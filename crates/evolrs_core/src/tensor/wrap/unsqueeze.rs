use crate::{
    device::Device,
    kind::Kind,
    shapes::shape::Shape,
    tensor::{RequiresGrad, Tensor},
};

pub trait Unsqueeze<const DIM: usize>: Shape {
    type UnsqueezeShape: Shape;
}

impl<S: Shape, D: Device, K: Kind, G: RequiresGrad> Tensor<S, D, K, G> {
    pub fn unsqueeze<const DIM: usize>(&self) -> Tensor<S::UnsqueezeShape, D, K, G>
    where
        S: Unsqueeze<DIM>,
    {
        Tensor {
            repr: self.repr.unsqueeze(DIM as i64),
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {}
