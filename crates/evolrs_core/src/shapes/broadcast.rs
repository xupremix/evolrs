use crate::{device::Device, kind::Kind, tensor::Tensor};

use super::shape::Shape;

pub trait Broadcast<Rhs: Shape, Dst: Shape>: Shape {
    const BROADCAST_CHECK: ();
}

pub trait BroadcastInplace<Rhs: Shape, Dst: Shape>: Shape {
    const BROADCAST_INPLACE_CHECK: ();
}

impl<Src: Shape, D: Device, K: Kind> Tensor<Src, D, K> {
    pub fn add<Dst: Shape, Rhs: Broadcast<Src, Dst>>(
        &self,
        other: &Tensor<Rhs, D, K>,
    ) -> Tensor<Dst, D, K> {
        #![allow(path_statements)]
        Rhs::BROADCAST_CHECK;
        Tensor {
            repr: self.repr.g_add(&other.repr),
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::shapes::broadcast::{Broadcast, BroadcastInplace};
    use crate::shapes::shape::{Rank1, Rank2};

    #[test]
    fn b_same_rank() {
        const _: () = <Rank1<20> as Broadcast<Rank1<20>, Rank1<20>>>::BROADCAST_CHECK;
        const _: () = <Rank2<10, 20> as Broadcast<Rank2<10, 20>, Rank2<10, 20>>>::BROADCAST_CHECK;
    }

    #[test]
    fn b_same_rank_1_value() {
        const _: () = <Rank1<20> as Broadcast<Rank1<1>, Rank1<20>>>::BROADCAST_CHECK;
        const _: () = <Rank1<1> as Broadcast<Rank1<20>, Rank1<20>>>::BROADCAST_CHECK;

        const _: () = <Rank2<10, 20> as Broadcast<Rank2<10, 1>, Rank2<10, 20>>>::BROADCAST_CHECK;
        const _: () = <Rank2<10, 20> as Broadcast<Rank2<1, 20>, Rank2<10, 20>>>::BROADCAST_CHECK;
        const _: () = <Rank2<10, 20> as Broadcast<Rank2<1, 1>, Rank2<10, 20>>>::BROADCAST_CHECK;
        const _: () = <Rank2<10, 1> as Broadcast<Rank2<10, 20>, Rank2<10, 20>>>::BROADCAST_CHECK;
        const _: () = <Rank2<1, 20> as Broadcast<Rank2<10, 20>, Rank2<10, 20>>>::BROADCAST_CHECK;
        const _: () = <Rank2<1, 1> as Broadcast<Rank2<10, 20>, Rank2<10, 20>>>::BROADCAST_CHECK;
    }

    #[test]
    fn b_different_rank() {
        const _: () = <Rank1<20> as Broadcast<Rank2<10, 20>, Rank2<10, 20>>>::BROADCAST_CHECK;
        const _: () = <Rank2<10, 20> as Broadcast<Rank1<20>, Rank2<10, 20>>>::BROADCAST_CHECK;
    }

    #[test]
    fn b_different_rank_1_value() {
        const _: () = <Rank1<20> as Broadcast<Rank2<10, 1>, Rank2<10, 20>>>::BROADCAST_CHECK;
        const _: () = <Rank1<1> as Broadcast<Rank2<10, 20>, Rank2<10, 20>>>::BROADCAST_CHECK;

        const _: () = <Rank2<10, 20> as Broadcast<Rank1<1>, Rank2<10, 20>>>::BROADCAST_CHECK;
        const _: () = <Rank2<10, 1> as Broadcast<Rank1<20>, Rank2<10, 20>>>::BROADCAST_CHECK;
    }

    #[test]
    fn bi_same_rank() {
        const _: () =
            <Rank1<20> as BroadcastInplace<Rank1<20>, Rank1<20>>>::BROADCAST_INPLACE_CHECK;
        const _: () = <Rank2<10, 20> as BroadcastInplace<Rank2<10, 20>, Rank2<10, 20>>>::BROADCAST_INPLACE_CHECK;
    }
}
