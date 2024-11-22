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
    // let a: Tensor<Rank2<2, 3>, Cpu, i32> = Tensor::ones();
    // let b = a + 2;
    let a: Tensor<Rank2<2, 3>, Cpu, i32> = Tensor::ones();
    let b = 2i64 + a;
    b.print();

    // let a = tch::Tensor::ones([2, 3], (tch::Kind::Int, tch::Device::Cpu));
    // let b = a + 2;
    let a = tch::Tensor::ones([2, 3], (tch::Kind::Int, tch::Device::Cpu));
    let b: tch::Tensor = 1099511600000000000i64 + a;
    b.print();
}
