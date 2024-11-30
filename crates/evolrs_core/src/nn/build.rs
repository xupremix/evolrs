use std::marker::PhantomData;

use tch::nn::Sequential;

use super::{vs::Vs, Model};

pub trait ModelBuilder: Sized {
    type Config;
    fn build(vs: &Vs, c: Self::Config) -> Model<Self> {
        let repr = tch::nn::seq();
        let repr = Self::step(vs, c, repr);
        Model {
            repr,
            module: PhantomData,
        }
    }

    fn step(vs: &Vs, c: Self::Config, seq: Sequential) -> Sequential;
}
