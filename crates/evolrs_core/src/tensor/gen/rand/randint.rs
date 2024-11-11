use crate::{device::Device, kind::IntOrFloat, shapes::shape::Shape, tensor::Tensor};

impl<S: Shape, D: Device, K: IntOrFloat> Tensor<S, D, K> {
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

    pub fn randint_like(&self, low: i64, high: i64) -> Self {
        Self {
            repr: self.repr.randint_like_low_dtype(low, high),
            ..Default::default()
        }
    }
}
