use evolrs::{
    device::Cpu,
    nn::{
        optim::{Backward, Sgd},
        vs::Vs,
    },
    shapes::shape::Tensor2,
    tensor::Grad,
};

fn main() {
    let vs: Vs = Vs::new();
    let mut sgd = Sgd::new(&vs, 0.01).unwrap();

    let r = Tensor2::<10, 10, Cpu, f32, Grad>::default();

    sgd.backward_step(&r.sum());
}
