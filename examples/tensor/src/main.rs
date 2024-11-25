use evolrs::{
    nn::{modules::linear::QuadLinear, Module},
    tch,
    tensor::Tensor3,
};

fn main() {
    let vs = tch::nn::VarStore::new(tch::Device::Cpu);
    let root = &vs.root();
    let qlin = QuadLinear::new(root / "qlin");
    let xs: Tensor3<30, 20, 10> = Tensor3::rand();
    let l = qlin.forward(&xs);
    l.print();
}
