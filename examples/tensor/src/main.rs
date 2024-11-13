use evolrs::{
    shapes::shape::{Rank2, Rank3},
    tensor::Tensor,
};

fn main() {
    let t1: Tensor<Rank2<4, 4>> = Tensor::rand();
    let t2: Tensor<Rank3<2, 1, 4>> = Tensor::rand();
    let t3: Tensor<Rank3<2, 4, 4>> = t1.add(&t2);
    t3.print();

    let t1: Tensor<Rank2<2, 3>> = Tensor::rand();
    let t2: Tensor<Rank2<4, 4>> = Tensor::rand();
    let t3 = t1.matmul(&t2);
}
