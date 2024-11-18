use evolrs::{
    device::{Cpu, Cuda},
    kind::c32,
    nn::{modules::linear::Linear, Module as _},
    shapes::shape::{Rank1, Rank2, Rank3},
    tch,
    tensor::{gen::full::Full, Tensor},
};

fn main() {
    // println!("{}", tch::Cuda::is_available());
    // let t: Tensor<Rank2<2, 5>, Cuda<0>, i32> = Tensor::ones();

    // Example of a layer forward
    // let vs = tch::nn::VarStore::new(tch::Device::Cpu);
    // let root = &vs.root();
    // let lin: Linear<5, 2> = Linear::new(root / "5x2", Default::default());
    // let out = lin.forward(t);
    // out.print();
    //
    // // Example of flattening
    // let t: Tensor<Rank3<1, 2, 2>, Cpu, c32> = Tensor::rand();
    // let flattened = t.flatten::<Rank1<4>>();
    // flattened.print();
    //
    // // Example of matmul
    // let t: Tensor<Rank3<2, 3, 4>, Cpu, c32> = Tensor::randn();
    // let t2: Tensor<Rank3<2, 4, 4>, _, _> = Tensor::randn();
    // let ris = t.matmul(&t2);
    // ris.print();
}
