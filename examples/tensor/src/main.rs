use evolrs::{shapes::shape::Rank2, tensor::Tensor};

fn main() {
    let t: Tensor<Rank2<6, 9>> = Tensor::ones();
    let ris = t.unsqueeze::<2>();
    ris.print();
}
