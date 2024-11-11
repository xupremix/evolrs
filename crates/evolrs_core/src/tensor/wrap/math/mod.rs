use crate::{device::Device, kind::Kind, shapes::shape::Shape, tensor::Tensor};

impl<S: Shape, D: Device, K: Kind> Tensor<S, D, K> {
    // abs
    // acos
    // arccos
    // addbmm If batch1 is a (b×n×m)(b×n×m) tensor, batch2 is a (b×m×p)(b×m×p) tensor, input must be broadcastable with a (n×p)(n×p) tensor and out will be a (n×p)(n×p) tensor.
    // add
    //
}
