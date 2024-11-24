use evolrs::tch::{self, nn::Module};

fn main() {
    let vs = tch::nn::VarStore::new(tch::Device::Cpu);
    let root = &vs.root();
    let lin = tch::nn::linear(root / "lin", 2, 3, Default::default());
    let xs = tch::Tensor::rand([4, 2], tch::kind::FLOAT_CPU);

    let xs = lin.forward(&xs);
    xs.print();
}
