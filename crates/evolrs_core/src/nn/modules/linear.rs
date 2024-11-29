use std::{borrow::Borrow, marker::PhantomData};

use tch::nn::{LinearConfig, Module as _, Sequential};

use crate::{
    device::{Cpu, Device},
    nn::{build::ModelBuilder, vs::Vs, Forward, Module},
    shapes::shape::Shape,
    tensor::{Grad, RequiresGrad, Tensor},
};

#[derive(Debug)]
pub struct Linear<const I: usize, const O: usize, D: Device = Cpu> {
    repr: tch::nn::Linear,
    device: PhantomData<D>,
}

impl<const I: usize, const O: usize, D: Device> Linear<I, O, D> {
    pub fn new<'a, V: Borrow<tch::nn::Path<'a>>>(vs: V, config: LinearConfig) -> Self {
        Self {
            repr: tch::nn::linear(vs, I as i64, O as i64, config),
            device: PhantomData,
        }
    }
}

impl<const I: usize, const O: usize, S: Forward<I, O>, D: Device, G: RequiresGrad>
    Module<Tensor<S, D, f32, G>> for Linear<I, O, D>
{
    type Output = Tensor<S::ForwardShape, D, f32, Grad>;

    fn forward(&self, xs: &Tensor<S, D, f32, G>) -> Self::Output {
        Tensor {
            repr: self.repr.forward(&xs.repr),
            ..Default::default()
        }
    }
}

impl<const I: usize, const O: usize, D: Device> ModelBuilder for Linear<I, O, D> {
    type Config = LinearConfig;

    fn step(vs: &Vs, c: Self::Config, seq: Sequential) -> Sequential {
        seq.add(tch::nn::linear(vs.root(), I as i64, O as i64, c))
    }
}

// pub struct QuadLinear {
//     l1: Linear<10, 20>,
//     l2: Linear<20, 30>,
//     l3: Linear<30, 20>,
//     l4: Linear<20, 10>,
// }
//
// impl QuadLinear {
//     pub fn new<'a, V>(vs: V) -> Self
//     where
//         V: Borrow<tch::nn::Path<'a>>,
//     {
//         let vs = vs.borrow();
//         Self {
//             l1: Linear::new(vs / "l1", Default::default()),
//             l2: Linear::new(vs / "l2", Default::default()),
//             l3: Linear::new(vs / "l3", Default::default()),
//             l4: Linear::new(vs / "l4", Default::default()),
//         }
//     }
// }
//
// impl<S: Shape, D: Device> Module<Tensor<S, D, f32>> for QuadLinear
// where
//     S: Forward<10, 20>,
//     S::ForwardShape: Forward<20, 30>,
//     <S::ForwardShape as Forward<20, 30>>::ForwardShape: Forward<30, 20>,
//     <<S::ForwardShape as Forward<20, 30>>::ForwardShape as Forward<30, 20>>::ForwardShape:
//         Forward<20, 10>,
//     Linear<10, 20>: Module<Tensor<S, D, f32>, Output = Tensor<S::ForwardShape, D, f32>>,
//     Linear<20, 30>: Module<
//         Tensor<S::ForwardShape, D, f32>,
//         Output = Tensor<<S::ForwardShape as Forward<20, 30>>::ForwardShape, D, f32>,
//     >,
//     Linear<30, 20>: Module<
//         Tensor<<S::ForwardShape as Forward<20, 30>>::ForwardShape, D, f32>,
//         Output = Tensor<<<S::ForwardShape as Forward<20, 30>>::ForwardShape as Forward<30, 20>>::ForwardShape, D, f32>,
//     >,
//     Linear<20, 10>: Module<
//         Tensor<<<S::ForwardShape as Forward<20, 30>>::ForwardShape as Forward<30, 20>>::ForwardShape, D, f32>,
//         Output = Tensor<<<<S::ForwardShape as Forward<20, 30>>::ForwardShape as Forward<30, 20>>::ForwardShape as Forward<20, 10>>::ForwardShape, D, f32>,
//     >,
// {
//     type Output = Tensor<<<<S::ForwardShape as Forward<20, 30>>::ForwardShape as Forward<30, 20>>::ForwardShape as Forward<20, 10>>::ForwardShape, D, f32>;
//
//     fn forward(&self, xs: &Tensor<S, D, f32>) -> Self::Output {
//         let xs = self.l1.forward(xs);
//         let xs = self.l2.forward(&xs);
//         let xs = self.l3.forward(&xs);
//         self.l4.forward(&xs)
//     }
// }
//

#[cfg(test)]
mod tests {}
