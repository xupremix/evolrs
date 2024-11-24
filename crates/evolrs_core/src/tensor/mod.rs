use std::marker::PhantomData;

pub mod gen;
pub mod wrap;

use crate::{
    device::{Cpu, Device},
    kind::Kind,
    shapes::shape::{Rank1, Rank2, Rank3, Rank4, Rank5, Rank6, Rank7, Rank8, Shape},
};

// Type aliases

pub type Tensor1<const D0: usize, D = Cpu, K = f32> = Tensor<Rank1<D0>, D, K>;
pub type Tensor2<const D0: usize, const D1: usize, D = Cpu, K = f32> = Tensor<Rank2<D0, D1>, D, K>;
pub type Tensor3<const D0: usize, const D1: usize, const D2: usize, D = Cpu, K = f32> =
    Tensor<Rank3<D0, D1, D2>, D, K>;
pub type Tensor4<
    const D0: usize,
    const D1: usize,
    const D2: usize,
    const D3: usize,
    D = Cpu,
    K = f32,
> = Tensor<Rank4<D0, D1, D2, D3>, D, K>;
pub type Tensor5<
    const D0: usize,
    const D1: usize,
    const D2: usize,
    const D3: usize,
    const D4: usize,
    D = Cpu,
    K = f32,
> = Tensor<Rank5<D0, D1, D2, D3, D4>, D, K>;
pub type Tensor6<
    const D0: usize,
    const D1: usize,
    const D2: usize,
    const D3: usize,
    const D4: usize,
    const D5: usize,
    D = Cpu,
    K = f32,
> = Tensor<Rank6<D0, D1, D2, D3, D4, D5>, D, K>;
pub type Tensor7<
    const D0: usize,
    const D1: usize,
    const D2: usize,
    const D3: usize,
    const D4: usize,
    const D5: usize,
    const D6: usize,
    D = Cpu,
    K = f32,
> = Tensor<Rank7<D0, D1, D2, D3, D4, D5, D6>, D, K>;
pub type Tensor8<
    const D0: usize,
    const D1: usize,
    const D2: usize,
    const D3: usize,
    const D4: usize,
    const D5: usize,
    const D6: usize,
    const D7: usize,
    D = Cpu,
    K = f32,
> = Tensor<Rank8<D0, D1, D2, D3, D4, D5, D6, D7>, D, K>;

#[must_use]
pub struct Tensor<S: Shape, D: Device = Cpu, K: Kind = f32> {
    pub(crate) repr: tch::Tensor,
    pub(crate) shape: PhantomData<S>,
    pub(crate) device: PhantomData<D>,
    pub(crate) dtype: PhantomData<K>,
}

impl<S: Shape, D: Device, K: Kind> Tensor<S, D, K> {
    pub const fn dims(&self) -> i64 {
        S::DIMS
    }

    pub const fn nelems(&self) -> usize {
        S::NELEMS
    }

    pub fn shape(&self) -> &[i64] {
        S::dims()
    }

    pub fn to_tch(&self) -> &tch::Tensor {
        &self.repr
    }

    pub fn to_tch_mut(&mut self) -> &mut tch::Tensor {
        &mut self.repr
    }

    pub fn print(&self) {
        self.repr.print();
    }
}

impl<S: Shape, D: Device, K: Kind> Default for Tensor<S, D, K> {
    fn default() -> Self {
        Self {
            repr: tch::Tensor::default(),
            shape: PhantomData,
            device: PhantomData,
            dtype: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {}
