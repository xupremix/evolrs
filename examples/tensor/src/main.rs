use evolrs::{
    shapes::shape::{Rank2, Rank3},
    tensor::Tensor,
};

fn main() {
    let mut t1: Tensor<Rank3<2, 3, 4>> = Tensor::ones();
    let t2: Tensor<Rank2<1, 1>> = Tensor::ones();
    let _ = t1.add_(&t2);
}
