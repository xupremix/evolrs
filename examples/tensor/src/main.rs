use evolrs::{shapes::shape::Rank3, tensor::Tensor};

fn main() {
    let t1: Tensor<Rank3<1, 2, 3>> = Tensor::ones();
    let t2: Tensor<Rank3<1, 3, 4>> = Tensor::ones();
    let t3 = t1.matmul(&t2);
    t3.print();
    // println!("Hello, world!");
}
