use evolrs::{device::Cpu, shapes::shape::Rank2, tch, tensor::Tensor};

// Rust traits in ops
// - ge
// - gt
// - le
// - lt
// - logical_and
// - logical_not
// - logical_or
// - logical_xor

fn main() {
    let t: Tensor<Rank2<2, 3>, Cpu, i32> = Tensor::ones();
    let t2: Tensor<Rank2<2, 3>, Cpu, i32> = Tensor::ones();
    let ris = (t * 4) << t2;
    ris.print();
}
