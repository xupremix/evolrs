use crate::device::Device;
use crate::kind::{restriction::composite::IntOrBool, scalar::IntoScalar, Kind};
use crate::shapes::shape::Shape;
use crate::tensor::Tensor;

macro_rules! def_fn {
    ($( $fn:ident $tch_fn:ident $fn_:ident $tch_fn_:ident $($restr:ident)?),* $(,)?) => {
        $(
            impl<S: Shape, D: Device, K: Kind $(+ $restr)?> Tensor<S, D, K> {
                pub fn $fn<T: IntoScalar $(+ $restr)?>(&self, rhs: T) -> Tensor<S, D, K>
                {
                    Tensor {
                        repr: self.repr.$tch_fn(rhs.into()),
                        ..Default::default()
                    }
                }
                pub fn $fn_<T: IntoScalar $(+ $restr)?>(&mut self, rhs: T) -> Tensor<S, D, K>
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

// NOTE: in torch operations with scalars do not have type promotion
// hence why adding a i64 to an i32 won't result in a i64 but in a i32
// (the resulting type is the same as the base tensor type)

def_fn! {
    add_s g_add_scalar add_s_ g_add_scalar_,
    sub_s g_sub_scalar sub_s_ g_sub_scalar_,
    mul_s g_mul_scalar mul_s_ g_mul_scalar_,
    div_s g_div_scalar div_s_ g_div_scalar_,
    bitwise_and bitwise_and bitwise_and_ bitwise_and_ IntOrBool,
    bitwise_or bitwise_or bitwise_or_ bitwise_or_ IntOrBool,
    bitwise_xor bitwise_xor bitwise_xor_ bitwise_xor_ IntOrBool,
    bitwise_left_shift bitwise_left_shift_tensor_scalar bitwise_left_shift_ bitwise_left_shift_tensor_scalar_ IntOrBool,
    bitwise_right_shift bitwise_right_shift_tensor_scalar bitwise_right_shift_ bitwise_right_shift_tensor_scalar_ IntOrBool,
}

#[cfg(test)]
mod tests {}
