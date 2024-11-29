use evolrs::{
    device::Cpu,
    nn::{optim::Sgd, vs::Vs},
    shapes::shape::Tensor2,
    tch::{self, nn::OptimizerConfig as _},
    tensor::{NoGrad, Uninitialized},
};

fn main() {
    // Example of Initalizer generic catching uninitialized tensor errors
    let vs: Vs = Vs::new();
    let mut sgd = Sgd::new(&vs, 0.01).unwrap();
    let r: Tensor2<2, 3, Cpu, f32, NoGrad, Uninitialized> = Tensor2::new();
    // let a = r.new_like().set_require_grad();
    // This will fail to compile

    let vs = tch::nn::VarStore::new(tch::Device::Cpu);
    let mut sgd = tch::nn::Sgd::default().build(&vs, 0.01).unwrap();
    let r = tch::Tensor::new().requires_grad_(true); // This will fail at runtime (undefined tensor)
}
