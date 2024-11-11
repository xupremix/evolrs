use evolrs::{
    device::Cpu,
    kind::{c16, c32, c64, f16},
    shapes::shape::{Rank1, Rank2, Rank3},
    tensor::{gen::logspace::LogspaceScalar, wrap::item::Item, Tensor},
};

fn main() {
    let t: Tensor<Rank3<1, 1, 1>, Cpu, f16> = Tensor::rand();
    let v = t.item();
    println!("{:?}", v);
    t.print();
}
