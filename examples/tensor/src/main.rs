use evolrs::{
    device::Cpu,
    kind::c32,
    shapes::shape::Rank1,
    tensor::{gen::logspace::LogspaceScalar, Tensor},
};

fn main() {
    let t1: Tensor<Rank1<5>, Cpu, c32> = Tensor::logspace(10, 20, 2.0);
    let t2: Tensor<Rank1<5>, Cpu, c32> = Tensor::logspace(10, 20, 2.0);
    let t3 = t1 + t2;
    t3.print();
    // println!("Hello, world!");
}
