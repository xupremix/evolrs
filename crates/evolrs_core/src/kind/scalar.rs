#[cfg(feature = "half")]
use super::f16;
use super::Kind;

pub trait IntoScalar: Kind {
    fn into(self) -> tch::Scalar;
}

macro_rules! impl_into_scalar {
    (@i64 $($t:ty),*) => {
        $(
            impl IntoScalar for $t {
                fn into(self) -> tch::Scalar {
                    tch::Scalar::from(self as i64)
                }
            }
        )*
    };
    (@f64 $($t:ty),*) => {
        $(
            impl IntoScalar for $t {
                fn into(self) -> tch::Scalar {
                    tch::Scalar::from(f64::from(self))
                }
            }
        )*
    };
}

impl_into_scalar!(@i64 u8, i8, i16, i32, i64);
impl_into_scalar!(@f64 f32, f64);

#[cfg(feature = "half")]
impl_into_scalar!(@f64 f16);
