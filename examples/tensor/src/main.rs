use evolrs::{shapes::shape::Rank1, tch, tensor::Tensor};

fn main() {
    let t: Tensor<Rank1<3>> = Tensor::randn();
    let argmax = t.argmax::<0, true>();
    argmax.print();

    let t = tch::Tensor::randn([3], tch::kind::FLOAT_CPU);
    let argmax = t.argmax(0, true);
    argmax.print();
}
