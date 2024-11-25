use evolrs::{
    nn::{modules::linear::Linear, Module},
    tch,
    tensor::Tensor3,
};

fn main() {
    let vs = tch::nn::VarStore::new(tch::Device::Cpu);
    let root = &vs.root();
    let lin: Linear<3, 4> = Linear::new(root / "lin", Default::default());
    let xs: Tensor3<1, 2, 3> = Tensor3::rand();

    let f = lin.forward(&xs);
    f.print();
}
