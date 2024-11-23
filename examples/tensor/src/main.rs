use evolrs::{device::Cpu, shapes::shape::Rank2, tch, tensor::Tensor};

fn main() {
    let t: Tensor<Rank2<2, 3>, Cpu, i32> = Tensor::ones();
    let ris = t.add_s(2.0);
    ris.print();
}
