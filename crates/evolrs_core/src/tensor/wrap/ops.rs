use crate::device::Device;
use crate::kind::{
    restriction::composite::IntOrBool,
    type_coercion::{Coerce, DivCoerce, Same},
    Kind,
};
use crate::shapes::shape::Shape;
use crate::tensor::Tensor;

#[cfg(feature = "broadcast-semantics")]
use crate::shapes::broadcast::{Broadcast, BroadcastInplace};

macro_rules! def_fn {
    ($( $kind:ident $trait:ident $fn:ident $tch_fn:ident $fn_:ident $tch_fn_:ident $($restr:ident)?),* $(,)?) => {
        $(
            impl<S: Shape, D: Device, K: $kind> Tensor<S, D, K> {
                #[cfg(feature = "broadcast-semantics")]
                pub fn $fn<Dst: Shape, Rhs: Broadcast<S, Dst>, K2: $trait<K> $(+ $restr)?>(
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
                pub fn $fn<K2: $trait<K> $(+ $restr)?>(&self, rhs: &Tensor<S, D, K2>) -> Tensor<S, D, K2::To> {
                    Tensor {
                        repr: self.repr.$tch_fn(&rhs.repr),
                        ..Default::default()
                    }
                }

                #[cfg(feature = "broadcast-semantics")]
                pub fn $fn_<Rhs: BroadcastInplace<S>, K2: $trait<K> $(+ $restr)?>(
                    &mut self,
                    rhs: &Tensor<Rhs, D, K2>,
                ) -> Tensor<S, D, K2::To>
                where
                    K: Same<K2::To>
                {
                    #![allow(path_statements)]
                    Rhs::BROADCAST_INPLACE_CHECK;
                    Tensor {
                        repr: self.repr.$tch_fn_(&rhs.repr),
                        ..Default::default()
                    }
                }

                #[cfg(not(feature = "broadcast-semantics"))]
                pub fn $fn_<K2: $trait<K> $(+ $restr)?>(
                    &mut self,
                    rhs: &Tensor<S, D, K2>,
                ) -> Tensor<S, D, K2::To>
                where
                    K: Same<K2::To>
                {
                    Tensor {
                        repr: self.repr.$tch_fn_(&rhs.repr),
                        ..Default::default()
                    }
                }
            }
        )*
    };
}

def_fn! {
    Kind Coerce add g_add add_ g_add_,
    Kind Coerce sub g_sub sub_ g_sub_,
    Kind Coerce mul g_mul mul_ g_mul_,
    Kind DivCoerce div g_div div_ g_div_,
    IntOrBool Coerce bitwise_and_t bitwise_and_tensor bitwise_and_t_ bitwise_and_tensor_ IntOrBool,
    IntOrBool Coerce bitwise_or_t bitwise_or_tensor bitwise_or_t_ bitwise_or_tensor_ IntOrBool,
    IntOrBool Coerce bitwise_xor_t bitwise_xor_tensor bitwise_xor_t_ bitwise_xor_tensor_ IntOrBool,
    IntOrBool Coerce bitwise_left_shift_t bitwise_left_shift bitwise_left_shift_t_ bitwise_left_shift_ IntOrBool,
    IntOrBool Coerce bitwise_right_shift_t bitwise_right_shift bitwise_right_shift_t_ bitwise_right_shift_ IntOrBool,
}

#[cfg(test)]
mod tests {}
