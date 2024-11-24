use evolrs::{
    nn::{modules::linear::Linear, Module},
    tch,
    tensor::Tensor2,
};

fn main() {
    let vs = tch::nn::VarStore::new(tch::Device::Cpu);
    let root = &vs.root();
    let lin: Linear<3, 9> = Linear::new(root / "lin", Default::default());
    let xs: Tensor2<6, 3> = Tensor2::rand();

    let f = lin.forward(&xs);
    f.print();
}
