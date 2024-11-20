use evolrs::{
    device::Cpu,
    kind::{c16, c64},
    shapes::shape::{Rank1, Rank2, Rank3},
    tch::{self},
    tensor::Tensor,
};

fn main() {
    let mut t1 = tch::Tensor::ones([2, 3, 4], tch::kind::FLOAT_CPU);
    let t2 = tch::Tensor::ones([1, 1], tch::kind::FLOAT_CPU);
    let _ = t1.g_add_(&t2);

    let mut t1: Tensor<Rank3<2, 3, 4>> = Tensor::ones();
    let t2: Tensor<Rank2<1, 1>> = Tensor::ones();
    let _ = t1.add_(&t2);
}
