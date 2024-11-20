use crate::device::Device;
use crate::kind::Kind;
use crate::shapes::shape::Shape;
use crate::tensor::Tensor;

#[cfg(feature = "broadcast-semantics")]
mod broadcast_add {
    use crate::shapes::broadcast::{Broadcast, BroadcastInplace};

    use super::*;
    impl<S: Shape, D: Device, K: Kind> Tensor<S, D, K> {
        pub fn add<Dst: Shape, Rhs: Broadcast<S, Dst>>(
            &self,
            rhs: &Tensor<Rhs, D, K>,
        ) -> Tensor<Dst, D, K> {
            #![allow(path_statements)]
            Rhs::BROADCAST_CHECK;
            Tensor {
                repr: self.repr.g_add(&rhs.repr),
                ..Default::default()
            }
        }

        pub fn add_<Rhs: BroadcastInplace<S>>(
            &mut self,
            rhs: &Tensor<Rhs, D, K>,
        ) -> Tensor<S, D, K> {
            #![allow(path_statements)]
            Rhs::BROADCAST_INPLACE_CHECK;
            Tensor {
                repr: self.repr.g_add_(&rhs.repr),
                ..Default::default()
            }
        }
    }
}

#[cfg(not(feature = "broadcast-semantics"))]
mod add {
    use super::*;
    impl<S: Shape, D: Device, K: Kind> Tensor<S, D, K> {}
}

// use crate::{device::Device, kind::Kind, tensor::Tensor};
// impl<Src: Shape, D: Device, K: Kind> Tensor<Src, D, K> {
//     pub fn add<Dst: Shape, Rhs: Broadcast<Src, Dst>>(
//         &self,
//         other: &Tensor<Rhs, D, K>,
//     ) -> Tensor<Dst, D, K> {
//         #![allow(path_statements)]
//         Rhs::BROADCAST_CHECK;
//         Tensor {
//             repr: self.repr.g_add(&other.repr),
//             ..Default::default()
//         }
//     }
// }
