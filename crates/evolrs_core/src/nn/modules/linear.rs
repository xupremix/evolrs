use std::{borrow::Borrow, marker::PhantomData};

use tch::nn::{LinearConfig, Module as _};

use crate::{
    device::{Cpu, Device},
    nn::{Forward, Module},
    shapes::shape::Shape,
    tensor::Tensor,
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

impl<const I: usize, const O: usize, S: Forward<I, O>, D: Device> Module<Tensor<S, D, f32>>
    for Linear<I, O, D>
{
    type Output = Tensor<S::ForwardShape, D, f32>;

    fn forward(&self, xs: &Tensor<S, D, f32>) -> Self::Output {
        Tensor {
            repr: self.repr.forward(&xs.repr),
            ..Default::default()
        }
    }
}

pub struct QuadLinear {
    l1: Linear<10, 20>,
    l2: Linear<20, 30>,
    l3: Linear<30, 20>,
    l4: Linear<20, 10>,
}

impl QuadLinear {
    pub fn new<'a, V>(vs: V) -> Self
    where
        V: Borrow<tch::nn::Path<'a>>,
    {
        let vs = vs.borrow();
        Self {
            l1: Linear::new(vs / "l1", Default::default()),
            l2: Linear::new(vs / "l2", Default::default()),
            l3: Linear::new(vs / "l3", Default::default()),
            l4: Linear::new(vs / "l4", Default::default()),
        }
    }
}

impl<S: Shape, D: Device> Module<Tensor<S, D, f32>> for QuadLinear
where
    S: Forward<10, 20>,
    S::ForwardShape: Forward<20, 30>,
    <S::ForwardShape as Forward<20, 30>>::ForwardShape: Forward<30, 20>,
    <<S::ForwardShape as Forward<20, 30>>::ForwardShape as Forward<30, 20>>::ForwardShape:
        Forward<20, 10>,
    Linear<10, 20>: Module<Tensor<S, D, f32>, Output = Tensor<S::ForwardShape, D, f32>>,
    Linear<20, 30>: Module<
        Tensor<S::ForwardShape, D, f32>,
        Output = Tensor<<S::ForwardShape as Forward<20, 30>>::ForwardShape, D, f32>,
    >,
    Linear<30, 20>: Module<
        Tensor<<S::ForwardShape as Forward<20, 30>>::ForwardShape, D, f32>,
        Output = Tensor<<<S::ForwardShape as Forward<20, 30>>::ForwardShape as Forward<30, 20>>::ForwardShape, D, f32>,
    >,
    Linear<20, 10>: Module<
        Tensor<<<S::ForwardShape as Forward<20, 30>>::ForwardShape as Forward<30, 20>>::ForwardShape, D, f32>,
        Output = Tensor<<<<S::ForwardShape as Forward<20, 30>>::ForwardShape as Forward<30, 20>>::ForwardShape as Forward<20, 10>>::ForwardShape, D, f32>,
    >,
{
    type Output = Tensor<<<<S::ForwardShape as Forward<20, 30>>::ForwardShape as Forward<30, 20>>::ForwardShape as Forward<20, 10>>::ForwardShape, D, f32>;

    fn forward(&self, xs: &Tensor<S, D, f32>) -> Self::Output {
        let xs = self.l1.forward(xs);
        let xs = self.l2.forward(&xs);
        let xs = self.l3.forward(&xs);
        self.l4.forward(&xs)
    }
}

impl<S, D, M0, M1, M2> Module<Tensor<S, D, f32>> for (M0, M1, M2)
where
    S: Shape,
    D: Device,
    M0: Module<Tensor<S, D, f32>>,
    M1: Module<M0::Output>,
    M2: Module<M1::Output>,
{
    type Output = M2::Output;

    fn forward(&self, xs: &Tensor<S, D, f32>) -> Self::Output {
        let xs = self.0.forward(xs);
        let xs = self.1.forward(&xs);
        self.2.forward(&xs)
    }
}

#[cfg(test)]
mod tests {}
