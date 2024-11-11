use std::{fmt::Debug, hash::Hash};

pub trait Kind: 'static + Debug + Clone + Copy + Send + Sync + PartialEq {
    fn into_dtype() -> tch::Kind;
}

macro_rules! kind {
    (def $n:ident $t:ident) => {
        #[allow(non_camel_case_types)]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub struct $n;
        kind!($n $t);
    };
    ($n:ident $t:ident) => {
        impl Kind for $n {
            fn into_dtype() -> tch::Kind {
                tch::Kind::$t
            }
        }
    };
}

kind!(u8 Uint8);
kind!(i8 Int8);
kind!(i16 Int16);
kind!(i32 Int);
kind!(i64 Int64);

#[cfg(feature = "half")]
pub use half::f16;
#[cfg(feature = "half")]
kind!(f16 Half);

kind!(f32 Float);
kind!(f64 Double);
kind!(bool Bool);

kind!(def c16 ComplexHalf);
kind!(def c32 ComplexFloat);
kind!(def c64 ComplexDouble);
kind!(def qi8 QInt8);
kind!(def qu8 QUInt8);
kind!(def qi32 QInt32);
kind!(def bf16 BFloat16);

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

pub trait IntOrFloat: Kind {}
macro_rules! impl_int_or_float {
    ($($t:ty),*) => {
        $(
            impl IntOrFloat for $t {}
        )*
    };
}
impl_int_or_float!(i8, i16, i32, i64, u8, f32, f64);
#[cfg(feature = "half")]
impl_int_or_float!(f16);

pub trait FloatOrComplex: Kind {}
macro_rules! impl_float_or_complex {
    ($($t:ty),*) => {
        $(
            impl FloatOrComplex for $t {}
        )*
    };
}
impl_float_or_complex!(f32, f64, c32, c16, c64);
#[cfg(feature = "half")]
impl_float_or_complex!(f16);

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! def_test {
        ($n:ident $t:ident $v:ident) => {
            #[test]
            fn $n() {
                assert_eq!($t::into_dtype(), tch::Kind::$v);
            }
        };
    }

    def_test!(testi8 i8 Int8);
    def_test!(testi16 i16 Int16);
    def_test!(testi32 i32 Int);
    def_test!(testi64 i64 Int64);

    #[cfg(feature = "half")]
    def_test!(testf16 f16 Half);

    def_test!(testf32 f32 Float);
    def_test!(testf64 f64 Double);
    def_test!(testbool bool Bool);

    def_test!(testc16 c16 ComplexHalf);
    def_test!(testc32 c32 ComplexFloat);
    def_test!(testc64 c64 ComplexDouble);
    def_test!(testqi8 qi8 QInt8);
    def_test!(testqu8 qu8 QUInt8);
    def_test!(testqi32 qi32 QInt32);
    def_test!(testbf16 bf16 BFloat16);
}
