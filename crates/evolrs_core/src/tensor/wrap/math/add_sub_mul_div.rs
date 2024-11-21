use crate::device::Device;
use crate::kind::type_coercion::{Coerce, DivCoerce, Same};
use crate::kind::Kind;
use crate::shapes::shape::Shape;
use crate::tensor::Tensor;

#[cfg(feature = "broadcast-semantics")]
use crate::shapes::broadcast::{Broadcast, BroadcastInplace};

macro_rules! def_fn {
    ($($trait:ident $fn:ident $tch_fn:ident $fn_:ident $tch_fn_:ident ),* $(,)?) => {
        $(
            impl<S: Shape, D: Device, K: Kind> Tensor<S, D, K> {
                #[cfg(feature = "broadcast-semantics")]
                pub fn $fn<Dst: Shape, Rhs: Broadcast<S, Dst>, K2: $trait<K>>(
                    &self,
                    rhs: &Tensor<Rhs, D, K2>,
                ) -> Tensor<Dst, D, K2::To> {
                    #![allow(path_statements)]
                    Rhs::BROADCAST_CHECK;
                    Tensor {
                        repr: self.repr.$tch_fn(&rhs.repr),
                        ..Default::default()
                    }
                }

                #[cfg(not(feature = "broadcast-semantics"))]
                pub fn $fn<K2: $trait<K>>(&self, rhs: &Tensor<S, D, K2>) -> Tensor<S, D, K2::To> {
                    Tensor {
                        repr: self.repr.$tch_fn(&rhs.repr),
                        ..Default::default()
                    }
                }

                #[cfg(feature = "broadcast-semantics")]
                pub fn $fn_<Rhs: BroadcastInplace<S>>(
                    &mut self,
                    rhs: &Tensor<Rhs, D, K>,
                ) -> Tensor<S, D, K> {
                    Tensor {
                        repr: self.repr.$tch_fn_(&rhs.repr),
                        ..Default::default()
                    }
                }

                #[cfg(not(feature = "broadcast-semantics"))]
                pub fn $fn_(&mut self, rhs: &Tensor<S, D, K>) -> Tensor<S, D, K> {
                    Tensor {
                        repr: self.repr.$tch_fn_(&rhs.repr),
                        ..Default::default()
                    }
                }
            }
        )*
    };
    (@div) => {
        impl<S: Shape, D: Device, K: Kind> Tensor<S, D, K> {
            #[cfg(feature = "broadcast-semantics")]
            pub fn div<Dst: Shape, Rhs: Broadcast<S, Dst>, K2: DivCoerce<K>>(
                &self,
                rhs: &Tensor<Rhs, D, K2>,
            ) -> Tensor<Dst, D, K2::To> {
                #![allow(path_statements)]
                Rhs::BROADCAST_CHECK;
                Tensor {
                    repr: self.repr.g_div(&rhs.repr),
                    ..Default::default()
                }
            }

            #[cfg(not(feature = "broadcast-semantics"))]
            pub fn div<K2: DivCoerce<K>>(&self, rhs: &Tensor<S, D, K2>) -> Tensor<S, D, K2::To> {
                Tensor {
                    repr: self.repr.g_div(&rhs.repr),
                    ..Default::default()
                }
            }

            #[cfg(feature = "broadcast-semantics")]
            pub fn div_<Rhs: BroadcastInplace<S>, K2: DivCoerce<K>>(
                &mut self,
                rhs: &Tensor<Rhs, D, K2>,
            ) -> Tensor<S, D, K2::To>
            where
                K: Same<K2::To>
            {
                Tensor {
                    repr: self.repr.g_div_(&rhs.repr),
                    ..Default::default()
                }
            }

            #[cfg(not(feature = "broadcast-semantics"))]
            pub fn div_<K2: DivCoerce<K>>(
                &mut self,
                rhs: &Tensor<S, D, K2>
            ) -> Tensor<S, D, K2::To>
            where
                K: Same<K2::To>
            {
                Tensor {
                    repr: self.repr.g_div_(&rhs.repr),
                    ..Default::default()
                }
            }
        }
    };
}

def_fn! {
    Coerce add g_add add_ g_add_,
    Coerce sub g_sub sub_ g_sub_,
    Coerce mul g_mul mul_ g_mul_,
}
def_fn! {
    @div
}
