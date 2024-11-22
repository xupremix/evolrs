use crate::kind::{
    scalar::IntoScalar,
    type_coercion::{Coerce, DivCoerce, Same},
};
use crate::{device::Device, shapes::shape::Shape, tensor::Tensor};

macro_rules! op {
    ($($coerce:ident $trait:ident $assign_trait:ident $method:ident $tch_method:ident $assign_method:ident $tch_assign_method:ident),* $(,)?) => {
        $(
            impl<T: IntoScalar, S: Shape, D: Device, K: $coerce<T>> std::ops::$trait<T> for Tensor<S, D, K> {
                type Output = Tensor<S, D, K::To>;

                fn $method(self, rhs: T) -> Self::Output {
                    Tensor {
                        repr: self.repr.$tch_method(rhs.into()),
                        ..Default::default()
                    }
                }
            }
            impl<T: IntoScalar, S: Shape, D: Device, K: $coerce<T>> std::ops::$trait<T> for &Tensor<S, D, K> {
                type Output = Tensor<S, D, K::To>;

                fn $method(self, rhs: T) -> Self::Output {
                    Tensor {
                        repr: self.repr.$tch_method(rhs.into()),
                        ..Default::default()
                    }
                }
            }
            impl<T: IntoScalar, S: Shape, D: Device, K: $coerce<T>> std::ops::$assign_trait<T>
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
                Coerce BitAnd bitand bitwise_and,
                Coerce BitOr bitor bitwise_or,
                Coerce BitXor bitxor bitwise_xor,
                Coerce Shl shl bitwise_left_shift_tensor_scalar,
                Coerce Shr shr bitwise_right_shift_tensor_scalar,
            }
        )*
    };
    (@impl $t:ty => $($coerce:ident $trait:ident $method:ident $tch_fn:ident),* $(,)? ) => {
        $(
            impl<S: Shape, D: Device, K: $coerce<$t>> std::ops::$trait<Tensor<S, D, K>> for $t {
                type Output = Tensor<S, D, K::To>;

                fn $method(self, rhs: Tensor<S, D, K>) -> Self::Output {
                    Tensor {
                        repr: rhs.repr.$tch_fn(<Self as IntoScalar>::into(self)),
                        ..Default::default()
                    }
                }
            }
            impl<S: Shape, D: Device, K: $coerce<$t>> std::ops::$trait<&Tensor<S, D, K>> for $t {
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
    Coerce BitAnd BitAndAssign bitand bitwise_and bitand_assign bitwise_and_,
    Coerce BitOr BitOrAssign bitor bitwise_or bitor_assign bitwise_or_,
    Coerce BitXor BitXorAssign bitxor bitwise_xor bitxor_assign bitwise_xor_,
    Coerce Shl ShlAssign shl bitwise_left_shift_tensor_scalar shl_assign bitwise_left_shift_tensor_scalar_,
    Coerce Shr ShrAssign shr bitwise_right_shift_tensor_scalar shr_assign bitwise_right_shift_tensor_scalar_,
}
op! {
    @inv u8 i8 i16 i32 i64 f32 f64
}
