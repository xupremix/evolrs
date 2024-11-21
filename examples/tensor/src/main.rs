use evolrs::{device::Cpu, kind::c32, shapes::shape::Rank2, tch, tensor::Tensor};

fn main() {
    let mut t1: Tensor<Rank2<2, 3>> = Tensor::ones();
    let t2: Tensor<Rank2<2, 3>, _, i64> = Tensor::ones();
    t1 += t2;
    t1.print();

    let mut t1 = tch::Tensor::ones([2, 3], tch::kind::FLOAT_CPU);
    let t2 = tch::Tensor::ones([2, 3], tch::kind::INT64_CPU);
    t1 += t2;
    t1.print();

    // let mut t1 = tch::Tensor::ones([2, 3], tch::kind::FLOAT_CPU);
    // let t2 = tch::Tensor::ones([2, 3], tch::kind::INT64_CPU);
    // t1 /= t2;
    // t1.print();
}
