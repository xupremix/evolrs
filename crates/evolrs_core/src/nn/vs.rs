use std::marker::PhantomData;

use crate::device::Device;

pub struct Vs<D: Device> {
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
}

impl<D: Device> Default for Vs<D> {
    fn default() -> Self {
        Self::new()
    }
}
