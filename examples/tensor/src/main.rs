use evolrs::{
    device::Cpu,
    kind::c32,
    shapes::shape::{Rank2, Rank3},
    tch,
    tensor::Tensor,
};

fn main() {
    let mut t1: Tensor<Rank2<2, 3>, Cpu, f64> = Tensor::ones();
    let t2: Tensor<Rank2<2, 3>, Cpu, f32> = Tensor::ones();
    let _ = t1.div_(&t2);

    // let mut t1 = tch::Tensor::ones([2, 3], tch::kind::FLOAT_CPU);
    // let t2 = tch::Tensor::ones([2, 3], tch::kind::INT64_CPU);
    // t1 /= t2;
    // t1.print();
}
