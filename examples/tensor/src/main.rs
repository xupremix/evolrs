use evolrs::{device::Cpu, kind::c16, shapes::shape::Rank2, tch, tensor::Tensor};

fn main() {
    let t1: Tensor<Rank2<2, 3>, Cpu, c16> = Tensor::ones();
    let t2: Tensor<Rank2<2, 3>> = Tensor::ones();
    let t3 = t2 + t1;
    t3.print();
}
