use crate::device::Device;
use crate::kind::{
    restriction::composite::IntOrBool,
    scalar::IntoScalar,
    type_coercion::{Coerce, DivCoerce},
    Kind,
};
use crate::shapes::shape::Shape;
use crate::tensor::{RequiresGrad, Tensor};
use crate::utils::Same;

macro_rules! def_fn {
    ($( $coerce:ident $fn:ident $tch_fn:ident $fn_:ident $tch_fn_:ident $($restr:ident)?),* $(,)?) => {
        $(
            impl<S: Shape, D: Device, K: Kind $(+ $restr)?, G: RequiresGrad> Tensor<S, D, K, G> {
                pub fn $fn<T: IntoScalar $(+ $restr)?>(&self, rhs: T) -> Tensor<S, D, K::To, G>
                where
                    K: $coerce<T>
                {
                    Tensor {
                        repr: self.repr.$tch_fn(rhs.into()),
                        ..Default::default()
                    }
                }
                pub fn $fn_<T: IntoScalar $(+ $restr)?>(&mut self, rhs: T) -> Tensor<S, D, K, G>
                where
                    K: $coerce<T> + Same<K::To>
                {
                    Tensor {
                        repr: self.repr.$tch_fn_(rhs.into()),
                        ..Default::default()
                    }
                }
            }
        )*
    };
}

def_fn! {
    Coerce add_s g_add_scalar add_s_ g_add_scalar_,
    Coerce sub_s g_sub_scalar sub_s_ g_sub_scalar_,
    Coerce mul_s g_mul_scalar mul_s_ g_mul_scalar_,
    DivCoerce div_s g_div_scalar div_s_ g_div_scalar_,
    Coerce bitwise_and bitwise_and bitwise_and_ bitwise_and_ IntOrBool,
    Coerce bitwise_or bitwise_or bitwise_or_ bitwise_or_ IntOrBool,
    Coerce bitwise_xor bitwise_xor bitwise_xor_ bitwise_xor_ IntOrBool,
    Coerce bitwise_left_shift bitwise_left_shift_tensor_scalar bitwise_left_shift_ bitwise_left_shift_tensor_scalar_ IntOrBool,
    Coerce bitwise_right_shift bitwise_right_shift_tensor_scalar bitwise_right_shift_ bitwise_right_shift_tensor_scalar_ IntOrBool,
}

#[cfg(test)]
mod tests {}
