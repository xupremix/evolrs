use std::marker::PhantomData;

use crate::device::{Cpu, Device};

pub struct Vs<D: Device = Cpu> {
    repr: tch::nn::VarStore,
    device: PhantomData<D>,
}

impl<D: Device> Vs<D> {
    pub fn new() -> Self {
        Self {
            repr: tch::nn::VarStore::new(D::into_device()),
            device: PhantomData,
        }
    }

    pub fn root(&self) -> tch::nn::Path {
        self.repr.root()
    }

    pub fn vs(&self) -> &tch::nn::VarStore {
        &self.repr
    }

    pub fn vs_mut(&mut self) -> &mut tch::nn::VarStore {
        &mut self.repr
    }
}

impl<D: Device> Default for Vs<D> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {}
