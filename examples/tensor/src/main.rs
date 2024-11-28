use evolrs::{
    nn::{
        optim::{Backward, Sgd},
        vs::Vs,
    },
    shapes::shape::Tensor1,
};

fn main() {
    let vs: Vs = Vs::new();
    let mut sgd = Sgd::new(&vs, 0.01).unwrap();

    let mut loss: Tensor1<10> = Tensor1::randn();
    let loss = loss.set_require_grad();
    sgd.backward_step(&loss.sum());
}
