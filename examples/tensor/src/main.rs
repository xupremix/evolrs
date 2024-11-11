use evolrs::{
    shapes::shape::{Rank2, Rank3},
    tensor::Tensor,
};

fn main() {
    let t: Tensor<Rank2<3, 2>> = Tensor::rand();
    let t2 = t.broadcast::<Rank3<4, 3, 2>>();
    t2.print();
}
