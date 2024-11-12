use evolrs::{
    shapes::shape::{Rank1, Rank3},
    tch,
    tensor::Tensor,
};

fn main() {
    let t1: Tensor<Rank1<3>> = Tensor::rand();
    let t2: Tensor<Rank3<2, 2, 3>> = Tensor::rand();
    let t3 = t1 + t2;
    t3.print();

    let t1 = tch::Tensor::rand([3], tch::kind::FLOAT_CPU);
    let t2 = tch::Tensor::rand([2, 2, 1], tch::kind::FLOAT_CPU);
    let t3 = t2 + t1;
    t3.print();
}
