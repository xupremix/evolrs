use evolrs::{device::Cpu, shapes::shape::Rank2, tensor::Tensor};

fn main() {
    let t: Tensor<Rank2<2, 3>, Cpu, i32> = Tensor::ones();
    let t2: Tensor<_, _, i16> = Tensor::ones();
    let ris = (t + t2 * 4).bitwise_left_shift(true);
    ris.print();
}
