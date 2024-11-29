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

impl<M0: ModelBuilder, M1: ModelBuilder, M2: ModelBuilder> ModelBuilder for (M0, M1, M2) {
    type Config = (M0::Config, M1::Config, M2::Config);

    fn step(vs: &Vs, c: Self::Config, seq: Sequential) -> Sequential {
        let (c0, c1, c2) = c;
        let seq = M0::step(vs, c0, seq);
        let seq = M1::step(vs, c1, seq);
        M2::step(vs, c2, seq)
    }
}
