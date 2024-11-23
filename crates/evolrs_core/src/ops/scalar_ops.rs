use crate::kind::{
    restriction::composite::IntOrBool,
    scalar::IntoScalar,
    type_coercion::{Coerce, DivCoerce, Same},
};
use crate::{device::Device, shapes::shape::Shape, tensor::Tensor};

macro_rules! op {
    ($($coerce:ident $trait:ident $assign_trait:ident $method:ident $tch_method:ident $assign_method:ident $tch_assign_method:ident $($restr:ident)?),* $(,)?) => {
        $(
            impl<T: IntoScalar $(+ $restr)?, S: Shape, D: Device, K: $coerce<T> $(+ $restr)?> std::ops::$trait<T> for Tensor<S, D, K> {
                type Output = Tensor<S, D, K::To>;

                fn $method(self, rhs: T) -> Self::Output {
                    Tensor {
                        repr: self.repr.$tch_method(rhs.into()),
                        ..Default::default()
                    }
                }
            }
            impl<T: IntoScalar $(+ $restr)?, S: Shape, D: Device, K: $coerce<T> $(+ $restr)?> std::ops::$trait<T> for &Tensor<S, D, K> {
                type Output = Tensor<S, D, K::To>;

                fn $method(self, rhs: T) -> Self::Output {
                    Tensor {
                        repr: self.repr.$tch_method(rhs.into()),
                        ..Default::default()
                    }
                }
            }
            impl<T: IntoScalar $(+ $restr)?, S: Shape, D: Device, K: $coerce<T> $(+ $restr)?> std::ops::$assign_trait<T>
                for Tensor<S, D, K>
            where
                K: Same<K::To>,
            {
                fn $assign_method(&mut self, rhs: T) {
                    let _ = self.repr.$tch_assign_method(rhs.into());
                }
            }
        )*
    };
    (@inv $($t:ty)*) => {
        $(
            op! {
                @impl $t =>
                Coerce Add add g_add_scalar,
                Coerce Sub sub g_sub_scalar,
                Coerce Mul mul g_mul_scalar,
                DivCoerce Div div g_div_scalar,
            }
        )*
    };
    (@restr $($t:ty)*) => {
        $(
            op! {
                @impl $t =>
                Coerce BitAnd bitand bitwise_and IntOrBool,
                Coerce BitOr bitor bitwise_or IntOrBool,
                Coerce BitXor bitxor bitwise_xor IntOrBool,
                Coerce Shl shl bitwise_left_shift_tensor_scalar IntOrBool,
                Coerce Shr shr bitwise_right_shift_tensor_scalar IntOrBool,
            }
        )*
    };
    (@impl $t:ty => $($coerce:ident $trait:ident $method:ident $tch_fn:ident $($restr:ident)?),* $(,)? ) => {
        $(
            impl<S: Shape, D: Device, K: $coerce<$t> $(+ $restr)?> std::ops::$trait<Tensor<S, D, K>> for $t {
                type Output = Tensor<S, D, K::To>;

                fn $method(self, rhs: Tensor<S, D, K>) -> Self::Output {
                    Tensor {
                        repr: rhs.repr.$tch_fn(<Self as IntoScalar>::into(self)),
                        ..Default::default()
                    }
                }
            }
            impl<S: Shape, D: Device, K: $coerce<$t> $(+ $restr)?> std::ops::$trait<&Tensor<S, D, K>> for $t {
                type Output = Tensor<S, D, K::To>;

                fn $method(self, rhs: &Tensor<S, D, K>) -> Self::Output {
                    Tensor {
                        repr: rhs.repr.$tch_fn(<Self as IntoScalar>::into(self)),
                        ..Default::default()
                    }
                }
            }
        )*
    };
}

op! {
    Coerce Add AddAssign add g_add_scalar add_assign g_add_scalar_,
    Coerce Sub SubAssign sub g_sub_scalar sub_assign g_sub_scalar_,
    Coerce Mul MulAssign mul g_mul_scalar mul_assign g_mul_scalar_,
    DivCoerce Div DivAssign div g_div_scalar div_assign g_div_scalar_,
    Coerce BitAnd BitAndAssign bitand bitwise_and bitand_assign bitwise_and_ IntOrBool,
    Coerce BitOr BitOrAssign bitor bitwise_or bitor_assign bitwise_or_ IntOrBool,
    Coerce BitXor BitXorAssign bitxor bitwise_xor bitxor_assign bitwise_xor_ IntOrBool,
    Coerce Shl ShlAssign shl bitwise_left_shift_tensor_scalar shl_assign bitwise_left_shift_tensor_scalar_ IntOrBool,
    Coerce Shr ShrAssign shr bitwise_right_shift_tensor_scalar shr_assign bitwise_right_shift_tensor_scalar_ IntOrBool,
}
op! {
    @inv u8 i8 i16 i32 i64 f32 f64
}
op! {
    @restr u8 i8 i16 i32 i64 bool
}
#[cfg(feature = "half")]
use crate::kind::f16;
#[cfg(feature = "half")]
op! {
    @inv f16
}
