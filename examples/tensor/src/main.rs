use evolrs::shapes::shape::Rank2;
use evolrs::{shapes::shape::Rank4, tensor::Tensor};

fn main() {
    let t: Tensor<Rank4<1, 6, 1, 9>> = Tensor::ones();
    let ris = t.squeeze_dim::<0>();
    ris.print();
}
