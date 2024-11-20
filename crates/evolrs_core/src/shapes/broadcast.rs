use super::shape::Shape;

pub trait Broadcast<Rhs: Shape, Dst: Shape>: Shape {
    const BROADCAST_CHECK: ();
}

pub trait BroadcastInplace<Src: Shape>: Shape {
    const BROADCAST_INPLACE_CHECK: ();
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
        const _: () = <Rank1<20> as BroadcastInplace<Rank1<20>>>::BROADCAST_INPLACE_CHECK;
        const _: () = <Rank2<10, 20> as BroadcastInplace<Rank2<10, 20>>>::BROADCAST_INPLACE_CHECK;
    }

    #[test]
    fn bi_same_rank_1_value() {
        const _: () = <Rank1<1> as BroadcastInplace<Rank1<20>>>::BROADCAST_INPLACE_CHECK;

        const _: () = <Rank2<10, 1> as BroadcastInplace<Rank2<10, 20>>>::BROADCAST_INPLACE_CHECK;
        const _: () = <Rank2<1, 20> as BroadcastInplace<Rank2<10, 20>>>::BROADCAST_INPLACE_CHECK;
        const _: () = <Rank2<1, 1> as BroadcastInplace<Rank2<10, 20>>>::BROADCAST_INPLACE_CHECK;
    }

    #[test]
    fn bi_different_rank() {
        const _: () = <Rank1<20> as BroadcastInplace<Rank2<10, 20>>>::BROADCAST_INPLACE_CHECK;
    }

    #[test]
    fn bi_different_rank_1_value() {
        const _: () = <Rank1<1> as BroadcastInplace<Rank2<10, 20>>>::BROADCAST_INPLACE_CHECK;
    }
}
