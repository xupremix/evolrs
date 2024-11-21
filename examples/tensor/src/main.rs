use evolrs::{shapes::shape::Rank2, tensor::Tensor};

fn main() {
    let t: Tensor<Rank2<2, 3>> = Tensor::ones();
    t.print();
}
