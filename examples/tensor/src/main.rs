use evolrs::{
    device::Cpu,
    nn::{modules::linear::Linear, optim::Sgd, Module},
    shapes::shape::Rank3,
    tch,
    tensor::Tensor3,
};

fn main() {
    let vs = tch::nn::VarStore::new(tch::Device::Cpu);
    let root = &vs.root();
    let lin: Linear<3, 4> = Linear::new(root / "lin", Default::default());
    let xs: Tensor3<1, 2, 3> = Tensor3::rand();
    let mut sgd: Sgd<Cpu> = Sgd::new(&vs, 0.01).unwrap();

    let l = lin.forward(&xs);

    sgd.backward::<Rank3<1, 2, 3>, Linear<3, 4>>(&l);
}
