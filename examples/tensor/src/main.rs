use evolrs::{device::Cpu, kind::c32, shapes::shape::Rank2, tch, tensor::Tensor};

// Rust traits in ops
// - bitwise_left_shift
// - bitwise_right_shift
// - ge
// - gt
// - le
// - lt
// - logical_and
// - logical_not
// - logical_or
// - logical_xor

fn main() {
    let a: Tensor<Rank2<2, 3>, Cpu, i64> = Tensor::ones();
    let b: Tensor<Rank2<2, 3>, _, bool> = Tensor::ones();
    let ris = b << a;
    ris.print();

    let a = tch::Tensor::ones([2, 3], tch::kind::INT64_CPU);
    let b = tch::Tensor::ones([2, 3], (tch::Kind::Bool, tch::Device::Cpu));
    let ris = a.bitwise_left_shift(&b);
    ris.print();
}
