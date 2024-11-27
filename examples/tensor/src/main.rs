// type Model = (
//     (Linear<10, 20>, Linear<20, 20>, Linear<20, 30>),
//     Linear<30, 20>,
//     Linear<20, 1>,
// );

// use evolrs::tch;
use evolrs::shapes::shape::Rank3;
use evolrs::tensor::Tensor2;

fn main() {
    let a: Tensor2<2, 3> = Tensor2::rand();
    let v = a.view::<Rank3<1, 1, 6>>();
    v.print();

    // let vs: Vs = Vs::new();
    // let root = &vs.root();
    // let qlin = QuadLinear::new(root / "qlin");
    // let xs: Tensor1<10> = Tensor1::rand();
    // let l = qlin.forward(&xs);
    // let mut sgd = Sgd::new(&vs, 0.01).unwrap();
    //
    // let model: Model = (
    //     (
    //         Linear::new(root / "a0", Default::default()),
    //         Linear::new(root / "a1", Default::default()),
    //         Linear::new(root / "a2", Default::default()),
    //     ),
    //     Linear::new(root / "b", Default::default()),
    //     Linear::new(root / "c", Default::default()),
    // );
    //
    // let loss = model.forward(&xs);
    // l.print();
    // loss.print();
    //
    // sgd.backward_step(&loss);
}
