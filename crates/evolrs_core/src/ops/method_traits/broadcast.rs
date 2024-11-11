use crate::{device::Device, kind::Kind, shapes::shape::Shape, tensor::Tensor};

pub trait Broadcast<T: Shape>: Shape {
    const CHECK: ();
    type BroadcastShape: Shape;
}

impl<Src: Shape, D: Device, K: Kind> Tensor<Src, D, K> {
    pub fn broadcast<Dst: Broadcast<Src>>(&self) -> Tensor<Dst::BroadcastShape, D, K> {
        #![allow(path_statements)]
        Dst::CHECK;
        Tensor {
            repr: self.repr.broadcast_to(Dst::dims()),
            ..Default::default()
        }
    }
}
