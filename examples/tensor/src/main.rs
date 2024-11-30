use evolrs::nn::optim::{Backward, Sgd};
use evolrs::nn::{build::ModelBuilder as _, modules::linear::Linear, vs::Vs, Module as _};
use evolrs::shapes::shape::Tensor2;

type Model = (
    (Linear<3, 5>, Linear<5, 10>, Linear<10, 20>),
    Linear<20, 40>,
    Linear<40, 10>,
);

fn main() {
    let vs: Vs = Vs::new();
    let mut sgd = Sgd::new(&vs, 0.01).unwrap();
    let model = Model::build(&vs, Default::default());
    let xs: Tensor2<5, 3> = Tensor2::rand();
    let loss = model.forward(&xs);
    sgd.backward_step(&loss.sum());

    model.print();
}
