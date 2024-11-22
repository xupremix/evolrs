use super::*;

pub trait Int: Kind {}
pub trait Float: Kind {}
pub trait Complex: Kind {}
pub trait Quant: Kind {}
pub trait Bool: Kind {}

macro_rules! impl_trait_for_kind {
        ($($n:ident: $($t:ty),*;)*) => {
            $(
            $(
            impl $n for $t {}
            )*
            )*
        };
    }

impl_trait_for_kind! {
    Int: u8, i8, i16, i32, i64;
    Float: f32, f64;
    Complex: c16, c32, c64;
    Quant: qi8, qu8, qi32, bf16;
    Bool: bool;
}

#[cfg(feature = "half")]
impl Float for f16 {}

pub mod composite {
    use super::*;

    macro_rules! def_composite {
            ($name:ident $($t:ty),+) => {
                pub trait $name: Kind {}
                $(
                   impl $name for $t {}
                )+
            };
        }

    #[cfg(feature = "half")]
    def_composite!(IntOrFloat i8, i16, i32, i64, u8, f16, f32, f64);
    #[cfg(not(feature = "half"))]
    def_composite!(IntOrFloat i8, i16, i32, i64, u8, f32, f64);

    #[cfg(feature = "half")]
    def_composite!(FloatOrComplex f16, f32, f64, c16, c32, c64);
    #[cfg(not(feature = "half"))]
    def_composite!(FloatOrComplex f32, f64, c16, c32, c64);

    def_composite!(IntOrBool i8, i16, i32, i64, u8, bool);

    #[cfg(feature = "half")]
    def_composite!(NotBool i8, i16, i32, i64, u8, f16, f32, f64, c16, c32, c64);
    #[cfg(not(feature = "half"))]
    def_composite!(NotBool i8, i16, i32, i64, u8, f32, f64, c16, c32, c64);
}
