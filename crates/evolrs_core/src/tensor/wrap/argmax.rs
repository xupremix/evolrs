use crate::{
    device::Device,
    kind::Kind,
    shapes::shape::Shape,
    tensor::{RequiresGrad, Tensor},
};

// Argmax
// keep_dim makes it so that the output tensor shape has the same number of
// dimensions, but the one selected will become 1.
//
// Example:
// let t = tch::Tensor::randn([4, 4, 4], tch::kind::FLOAT_CPU);
// // t.argmax(1, false) is a 4x1x4 tensor
//
// dim must be in the range [0, S::DIMS)
//
// the output tensor will either have the same dimensions as the input tensor
// (with the selected dimension converted to 1)
// or it will be a Rank1 tensor with N = S::DIMS

pub trait ArgmaxShape<const DIM: i64, const KEEP_DIM: bool>: Shape {
    type ArgmaxShape: Shape;
}

impl<S: Shape, D: Device, K: Kind, G: RequiresGrad> Tensor<S, D, K, G> {
    pub fn argmax<const DIM: i64, const KEEP_DIM: bool>(&self) -> Tensor<S::ArgmaxShape, D, K>
    where
        S: ArgmaxShape<DIM, KEEP_DIM>,
    {
        Tensor {
            repr: self.repr.argmax(DIM, KEEP_DIM),
            ..Default::default()
        }
    }
}
