use evolrs::{
    nn::{
        modules::linear::{Linear, QuadLinear},
        Module,
    },
    tch,
    tensor::Tensor3,
};

type Model = (Linear<10, 20>, Linear<20, 20>, Linear<20, 10>);

fn main() {
    let vs = tch::nn::VarStore::new(tch::Device::Cpu);
    let root = &vs.root();
    let qlin = QuadLinear::new(root / "qlin");
    let xs: Tensor3<30, 20, 10> = Tensor3::rand();
    let l = qlin.forward(&xs);

    let model: Model = (
        Linear::new(root / "a", Default::default()),
        Linear::new(root / "b", Default::default()),
        Linear::new(root / "c", Default::default()),
    );

    let out = model.forward(&xs);
    l.print();
    out.print();
}
