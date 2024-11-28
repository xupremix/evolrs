use evolrs::shapes::shape::Tensor2;

fn main() {
    let t: Tensor2<10, 20> = Tensor2::rand();
    t.print();
}
