use evolrs::nn::{modules::linear::Linear, vs::Vs, Module};
use evolrs::shapes::shape::Rank2;
use evolrs::tensor::Tensor;

type Custom = (Linear<3, 20>, Linear<20, 20>, Linear<20, 4>);

fn main() {
    let vs: Vs = Vs::new();
    type Forward = Tensor<Rank2<5, 3>>;
    let model = <Custom as Module<Forward>>::build(&vs, Default::default());

    let xs: Forward = Tensor::rand();
    let out = model.forward(&xs);
    out.print();
}
