// - bernoulli
// - uniform
// - cauchy
// - geometric
// - log_normal
// - normal_
// - random_

// use crate::{
//     device::Device,
//     kind::{restriction::composite::FloatOrComplex, Kind},
//     shapes::shape::Shape,
//     tensor::Tensor,
// };
//
// impl<S: Shape, D: Device, K: FloatOrComplex> Tensor<S, D, K> {
//     pub fn bernoulli() -> Self {
//         // let mut out = Tensor::randn();
//         // out.repr = out.repr.bernoulli();
//         // out
//         let out = Self::rand();
//         Self {
//             repr: out.repr.bernoulli(),
//             ..Default::default()
//         }
//     }
//     pub fn bernoulli_p(p: f64) -> Self {
//         let mut out = Self::empty();
//         out.repr = out.repr.bernoulli_p(p);
//         out
//     }
//     pub fn bernoulli_like(&self) -> Self {
//         Self {
//             repr: self.repr.bernoulli(),
//             ..Default::default()
//         }
//     }
//     pub fn bernoulli_like_p(&self, p: f64) -> Self {
//         Self {
//             repr: self.repr.bernoulli_p(p),
//             ..Default::default()
//         }
//     }
//     pub fn bernoulli_(&mut self) -> Self {
//         self.repr = self.repr.bernoulli();
//         Self {
//             repr: self.repr.copy(),
//             ..Default::default()
//         }
//     }
//     pub fn bernoulli_p_(&mut self, p: f64) -> Self {
//         self.repr = self.repr.bernoulli_p(p);
//         Self {
//             repr: self.repr.copy(),
//             ..Default::default()
//         }
//     }
// }
