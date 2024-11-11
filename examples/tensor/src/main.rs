use evolrs::{
    device::Cpu,
    kind::{c16, c32, c64},
    shapes::shape::{Rank1, Rank2},
    tensor::{gen::logspace::LogspaceScalar, Tensor},
};

fn main() {
    let t: Tensor<Rank1<2>, Cpu, u8> = Tensor::randint(0, 10);
    t.print();
}
