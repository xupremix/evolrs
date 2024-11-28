use crate::{
    device::Device,
    kind::restriction::composite::IntOrFloat,
    shapes::shape::Shape,
    tensor::{NoGrad, RequiresGrad, Tensor},
};

impl<S: Shape, D: Device, K: IntOrFloat> Tensor<S, D, K, NoGrad> {
    pub fn randint(low: i64, high: i64) -> Self {
        Self {
            repr: tch::Tensor::randint_low(
                low,
                high,
                S::dims(),
                (K::into_dtype(), D::into_device()),
            ),
            ..Default::default()
        }
    }
}
impl<S: Shape, D: Device, K: IntOrFloat, G: RequiresGrad> Tensor<S, D, K, G> {
    pub fn randint_like(&self, low: i64, high: i64) -> Self {
        Self {
            repr: self.repr.randint_like_low_dtype(low, high),
            ..Default::default()
        }
    }
}
