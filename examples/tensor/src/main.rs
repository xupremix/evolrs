use evolrs::{
    device::Cpu,
    kind::c32,
    shapes::shape::{Rank2, Rank3},
    tensor::Tensor,
};

fn main() {
    let t1: Tensor<Rank3<2, 3, 4>, Cpu, f64> = Tensor::ones();
    let t2: Tensor<Rank2<1, 1>, _, c32> = Tensor::ones();
    let t3: Tensor<Rank3<2, 3, 4>, _, _> = t1.add(&t2);
    t3.print();
}
