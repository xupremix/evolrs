use evolrs::{
    nn::{modules::linear::Linear, Module as _},
    shapes::shape::Rank2,
    tch,
    tensor::Tensor,
};

fn main() {
    let t: Tensor<Rank2<10, 5>> = Tensor::rand();
    let vs = tch::nn::VarStore::new(tch::Device::Cpu);
    let root = &vs.root();
    let lin: Linear<5, 2> = Linear::new(root / "10x2", Default::default());
    let out = lin.forward(t);
    out.print();
}
