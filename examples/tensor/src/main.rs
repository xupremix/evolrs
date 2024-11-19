use evolrs::{device::Cpu, shapes::shape::Rank2, tensor::Tensor};

fn main() {
    let t: Tensor<Rank2<2, 3>, Cpu, bool> = Tensor::eye();
    t.print();
}
