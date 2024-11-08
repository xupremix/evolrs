use evolrs::{
    shapes::shape::{Rank1, Rank3},
    tensor::Tensor,
};

fn main() {
    let t1: Tensor<Rank3<1, 2, 3>> = Tensor::ones();
    let t2: Tensor<Rank1<6>> = t1.flatten();
    let t2 = t2.flatten::<Rank1<6>>();
    // println!("Hello, world!");
}
