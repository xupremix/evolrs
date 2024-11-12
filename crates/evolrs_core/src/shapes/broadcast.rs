use super::shape::Shape;

pub trait Broadcast<T: Shape>: Shape {
    const CHECK: ();
    type BroadcastShape: Shape;
}

#[cfg(test)]
mod tests {
    use crate::shapes::broadcast::Broadcast;
    use crate::shapes::shape::{Rank1, Rank3};

    #[test]
    fn same_rank() {
        const _: () = <Rank1<20> as Broadcast<Rank1<20>>>::CHECK;
        const _: () = <Rank3<10, 20, 30> as Broadcast<Rank3<10, 20, 30>>>::CHECK;
    }

    #[test]
    fn same_rank_1_value() {
        const _: () = <Rank1<20> as Broadcast<Rank1<1>>>::CHECK;
        const _: () = <Rank1<1> as Broadcast<Rank1<20>>>::CHECK;

        const _: () = <Rank3<10, 20, 30> as Broadcast<Rank3<10, 20, 1>>>::CHECK;
        const _: () = <Rank3<10, 20, 30> as Broadcast<Rank3<10, 1, 30>>>::CHECK;
        const _: () = <Rank3<10, 20, 30> as Broadcast<Rank3<10, 1, 1>>>::CHECK;
        const _: () = <Rank3<10, 20, 30> as Broadcast<Rank3<1, 20, 30>>>::CHECK;
        const _: () = <Rank3<10, 20, 30> as Broadcast<Rank3<1, 20, 1>>>::CHECK;
        const _: () = <Rank3<10, 20, 30> as Broadcast<Rank3<1, 1, 30>>>::CHECK;
        const _: () = <Rank3<10, 20, 30> as Broadcast<Rank3<1, 1, 1>>>::CHECK;

        const _: () = <Rank3<10, 20, 1> as Broadcast<Rank3<10, 20, 30>>>::CHECK;
        const _: () = <Rank3<10, 1, 30> as Broadcast<Rank3<10, 20, 30>>>::CHECK;
        const _: () = <Rank3<10, 1, 1> as Broadcast<Rank3<10, 20, 30>>>::CHECK;
        const _: () = <Rank3<1, 20, 30> as Broadcast<Rank3<10, 20, 30>>>::CHECK;
        const _: () = <Rank3<1, 20, 1> as Broadcast<Rank3<10, 20, 30>>>::CHECK;
        const _: () = <Rank3<1, 1, 30> as Broadcast<Rank3<10, 20, 30>>>::CHECK;
        const _: () = <Rank3<1, 1, 1> as Broadcast<Rank3<10, 20, 30>>>::CHECK;
    }

    #[test]
    fn different_rank() {
        const _: () = <Rank1<30> as Broadcast<Rank3<10, 20, 30>>>::CHECK;
        const _: () = <Rank3<10, 20, 30> as Broadcast<Rank1<30>>>::CHECK;
    }

    #[test]
    fn different_rank_1_value() {
        const _: () = <Rank1<30> as Broadcast<Rank3<10, 20, 1>>>::CHECK;
        const _: () = <Rank1<30> as Broadcast<Rank3<10, 1, 30>>>::CHECK;
        const _: () = <Rank1<30> as Broadcast<Rank3<10, 1, 1>>>::CHECK;
        const _: () = <Rank1<30> as Broadcast<Rank3<1, 20, 30>>>::CHECK;
        const _: () = <Rank1<30> as Broadcast<Rank3<1, 20, 1>>>::CHECK;
        const _: () = <Rank1<30> as Broadcast<Rank3<1, 1, 30>>>::CHECK;
        const _: () = <Rank1<30> as Broadcast<Rank3<1, 1, 1>>>::CHECK;

        const _: () = <Rank3<10, 20, 1> as Broadcast<Rank1<30>>>::CHECK;
        const _: () = <Rank3<10, 1, 30> as Broadcast<Rank1<30>>>::CHECK;
        const _: () = <Rank3<10, 1, 1> as Broadcast<Rank1<30>>>::CHECK;
        const _: () = <Rank3<1, 20, 30> as Broadcast<Rank1<30>>>::CHECK;
        const _: () = <Rank3<1, 20, 1> as Broadcast<Rank1<30>>>::CHECK;
        const _: () = <Rank3<1, 1, 30> as Broadcast<Rank1<30>>>::CHECK;
        const _: () = <Rank3<1, 1, 1> as Broadcast<Rank1<30>>>::CHECK;
    }
}
