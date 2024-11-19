use crate::{
    device::Device,
    kind::{c16, c32, c64},
    shapes::shape::Shape,
    tensor::Tensor,
};

macro_rules! operator {
    ($trait:ident $method:ident $tch_method:ident => $from:ty => $rhs:ty => $to:ty) => {
        impl<S: Shape, D: Device> std::ops::$trait<Tensor<S, D, $rhs>> for Tensor<S, D, $from> {
            type Output = Tensor<S, D, $to>;
            fn $method(self, rhs: Tensor<S, D, $rhs>) -> Self::Output {
                Tensor {
                    repr: self.repr.$tch_method(&rhs.repr),
                    ..Default::default()
                }
            }
        }
        impl<S: Shape, D: Device> std::ops::$trait<&Tensor<S, D, $rhs>> for Tensor<S, D, $from> {
            type Output = Tensor<S, D, $to>;
            fn $method(self, rhs: &Tensor<S, D, $rhs>) -> Self::Output {
                Tensor {
                    repr: self.repr.$tch_method(&rhs.repr),
                    ..Default::default()
                }
            }
        }
        impl<S: Shape, D: Device> std::ops::$trait<&Tensor<S, D, $rhs>> for &Tensor<S, D, $from> {
            type Output = Tensor<S, D, $to>;

            fn $method(self, rhs: &Tensor<S, D, $rhs>) -> Self::Output {
                Tensor {
                    repr: self.repr.$tch_method(&rhs.repr),
                    ..Default::default()
                }
            }
        }
        impl<S: Shape, D: Device> std::ops::$trait<Tensor<S, D, $rhs>> for &Tensor<S, D, $from> {
            type Output = Tensor<S, D, $to>;

            fn $method(self, rhs: Tensor<S, D, $rhs>) -> Self::Output {
                Tensor {
                    repr: self.repr.$tch_method(&rhs.repr),
                    ..Default::default()
                }
            }
        }
    };
}

macro_rules! def_op {
    ($modn:ident $trait:ident $method:ident $tch_method:ident) => {
        operator!($trait $method $tch_method => bool => bool => bool);

        operator!($trait $method $tch_method => i8 => i8 => i8);
        operator!($trait $method $tch_method => i8 => bool => i8);
        operator!($trait $method $tch_method => bool => i8 => i8);

        operator!($trait $method $tch_method => u8 => u8 => u8);
        operator!($trait $method $tch_method => u8 => bool => u8);
        operator!($trait $method $tch_method => bool => u8 => u8);

        /*
        Uint8 + Int8 = Int16
        Int8 + Uint8 = Int16
        Uint8 + Int16 = Int16
        Int16 + Uint8 = Int16
        Int8 + Int16 = Int16
        Int16 + Int8 = Int16
        Int16 + Int16 = Int16
        Int16 + Bool = Int16
        Bool + Int16 = Int16
        */
        operator!($trait $method $tch_method => u8 => i8 => i16);
        operator!($trait $method $tch_method => i8 => u8 => i16);
        operator!($trait $method $tch_method => u8 => i16 => i16);
        operator!($trait $method $tch_method => i16 => u8 => i16);
        operator!($trait $method $tch_method => i8 => i16 => i16);
        operator!($trait $method $tch_method => i16 => i8 => i16);
        operator!($trait $method $tch_method => i16 => i16 => i16);
        operator!($trait $method $tch_method => i16 => bool => i16);
        operator!($trait $method $tch_method => bool => i16 => i16);

        /*
        Int + Int = Int
        Uint8 + Int = Int
        Int + Uint8 = Int
        Int8 + Int = Int
        Int + Int8 = Int
        Int16 + Int = Int
        Int + Int16 = Int
        Int + Bool = Int
        Bool + Int = Int
        */
        operator!($trait $method $tch_method => i32 => i32 => i32);
        operator!($trait $method $tch_method => u8 => i32 => i32);
        operator!($trait $method $tch_method => i32 => u8 => i32);
        operator!($trait $method $tch_method => i8 => i32 => i32);
        operator!($trait $method $tch_method => i32 => i8 => i32);
        operator!($trait $method $tch_method => i16 => i32 => i32);
        operator!($trait $method $tch_method => i32 => i16 => i32);
        operator!($trait $method $tch_method => i32 => bool => i32);
        operator!($trait $method $tch_method => bool => i32 => i32);

        /*
        Int64 + Int64 = Int64
        Uint8 + Int64 = Int64
        Int64 + Uint8 = Int64
        Int8 + Int64 = Int64
        Int64 + Int8 = Int64
        Int16 + Int64 = Int64
        Int64 + Int16 = Int64
        Int + Int64 = Int64
        Int64 + Int = Int64
        Int64 + Bool = Int64
        Bool + Int64 = Int64
        */
        operator!($trait $method $tch_method => i64 => i64 => i64);
        operator!($trait $method $tch_method => u8 => i64 => i64);
        operator!($trait $method $tch_method => i64 => u8 => i64);
        operator!($trait $method $tch_method => i8 => i64 => i64);
        operator!($trait $method $tch_method => i64 => i8 => i64);
        operator!($trait $method $tch_method => i16 => i64 => i64);
        operator!($trait $method $tch_method => i64 => i16 => i64);
        operator!($trait $method $tch_method => i32 => i64 => i64);
        operator!($trait $method $tch_method => i64 => i32 => i64);
        operator!($trait $method $tch_method => i64 => bool => i64);
        operator!($trait $method $tch_method => bool => i64 => i64);

        /*
        Half + Half = Half
        Uint8 + Half = Half
        Half + Uint8 = Half
        Int8 + Half = Half
        Half + Int8 = Half
        Int16 + Half = Half
        Half + Int16 = Half
        Int + Half = Half
        Half + Int = Half
        Int64 + Half = Half
        Half + Int64 = Half
        Bool + Half = Half
        Half + Bool = Half

        Half + Float = Float
        Float + Half = Float

        Half + Double = Double
        Double + Half = Double

        Half + ComplexHalf = ComplexHalf
        ComplexHalf + Half = ComplexHalf

        Half + ComplexFloat = ComplexFloat
        ComplexFloat + Half = ComplexFloat

        Half + ComplexDouble = ComplexDouble
        ComplexDouble + Half = ComplexDouble
        */
        #[cfg(feature = "half")]
        mod $modn {
            use super::*;
            use crate::kind::{c16, c32, c64, f16};

            operator!($trait $method $tch_method => f16 => f16 => f16);
            operator!($trait $method $tch_method => u8 => f16 => f16);
            operator!($trait $method $tch_method => f16 => u8 => f16);
            operator!($trait $method $tch_method => i8 => f16 => f16);
            operator!($trait $method $tch_method => f16 => i16 => f16);
            operator!($trait $method $tch_method => i32 => f16 => f16);
            operator!($trait $method $tch_method => f16 => i32 => f16);
            operator!($trait $method $tch_method => i64 => f16 => f16);
            operator!($trait $method $tch_method => f16 => i64 => f16);
            operator!($trait $method $tch_method => bool => f16 => f16);
            operator!($trait $method $tch_method => f16 => bool => f16);

            operator!($trait $method $tch_method => f16 => f32 => f32);
            operator!($trait $method $tch_method => f32 => f16 => f32);

            operator!($trait $method $tch_method => f16 => f64 => f64);
            operator!($trait $method $tch_method => f64 => f16 => f64);

            operator!($trait $method $tch_method => f16 => c16 => c16);
            operator!($trait $method $tch_method => c16 => f16 => c16);

            operator!($trait $method $tch_method => f16 => c32 => c32);
            operator!($trait $method $tch_method => c32 => f16 => c32);

            operator!($trait $method $tch_method => f16 => c64 => c64);
            operator!($trait $method $tch_method => c64 => f16 => c64);
        }

        /*
        Float + Float = Float
        Uint8 + Float = Float
        Float + Uint8 = Float
        Int8 + Float = Float
        Float + Int8 = Float
        Int16 + Float = Float
        Float + Int16 = Float
        Int + Float = Float
        Float + Int = Float
        Int64 + Float = Float
        Float + Int64 = Float
        Float + Bool = Float
        Bool + Float = Float
        */
        operator!($trait $method $tch_method => f32 => f32 => f32);
        operator!($trait $method $tch_method => u8 => f32 => f32);
        operator!($trait $method $tch_method => f32 => u8 => f32);
        operator!($trait $method $tch_method => i8 => f32 => f32);
        operator!($trait $method $tch_method => f32 => i8 => f32);
        operator!($trait $method $tch_method => i16 => f32 => f32);
        operator!($trait $method $tch_method => f32 => i16 => f32);
        operator!($trait $method $tch_method => i32 => f32 => f32);
        operator!($trait $method $tch_method => f32 => i32 => f32);
        operator!($trait $method $tch_method => i64 => f32 => f32);
        operator!($trait $method $tch_method => f32 => i64 => f32);
        operator!($trait $method $tch_method => bool => f32 => f32);
        operator!($trait $method $tch_method => f32 => bool => f32);

        /*
        Double + Double = Double
        Int8 + Double = Double
        Double + Int8 = Double
        Uint8 + Double = Double
        Double + Uint8 = Double
        Int16 + Double = Double
        Double + Int16 = Double
        Int + Double = Double
        Double + Int = Double
        Int64 + Double = Double
        Double + Int64 = Double
        Float + Double = Double
        Double + Float = Double
        Bool + Double = Double
        Double + Bool = Double
        */
        operator!($trait $method $tch_method => f64 => f64 => f64);
        operator!($trait $method $tch_method => i8 => f64 => f64);
        operator!($trait $method $tch_method => f64 => i8 => f64);
        operator!($trait $method $tch_method => u8 => f64 => f64);
        operator!($trait $method $tch_method => f64 => u8 => f64);
        operator!($trait $method $tch_method => i16 => f64 => f64);
        operator!($trait $method $tch_method => f64 => i16 => f64);
        operator!($trait $method $tch_method => i32 => f64 => f64);
        operator!($trait $method $tch_method => f64 => i32 => f64);
        operator!($trait $method $tch_method => i64 => f64 => f64);
        operator!($trait $method $tch_method => f64 => i64 => f64);
        operator!($trait $method $tch_method => f32 => f64 => f64);
        operator!($trait $method $tch_method => f64 => f32 => f64);
        operator!($trait $method $tch_method => bool => f64 => f64);
        operator!($trait $method $tch_method => f64 => bool => f64);

        /*
        ComplexHalf + ComplexHalf = ComplexHalf
        Uint8 + ComplexHalf = ComplexHalf
        ComplexHalf + Uint8 = ComplexHalf
        Int8 + ComplexHalf = ComplexHalf
        ComplexHalf + Int8 = ComplexHalf
        Int16 + ComplexHalf = ComplexHalf
        ComplexHalf + Int16 = ComplexHalf
        Int + ComplexHalf = ComplexHalf
        ComplexHalf + Int = ComplexHalf
        Int64 + ComplexHalf = ComplexHalf
        ComplexHalf + Int64 = ComplexHalf
        Bool + ComplexHalf = ComplexHalf
        ComplexHalf + Bool = ComplexHalf
        */
        operator!($trait $method $tch_method => c16 => c16 => c16);
        operator!($trait $method $tch_method => u8 => c16 => c16);
        operator!($trait $method $tch_method => c16 => u8 => c16);
        operator!($trait $method $tch_method => i8 => c16 => c16);
        operator!($trait $method $tch_method => c16 => i8 => c16);
        operator!($trait $method $tch_method => i16 => c16 => c16);
        operator!($trait $method $tch_method => c16 => i16 => c16);
        operator!($trait $method $tch_method => i32 => c16 => c16);
        operator!($trait $method $tch_method => c16 => i32 => c16);
        operator!($trait $method $tch_method => i64 => c16 => c16);
        operator!($trait $method $tch_method => c16 => i64 => c16);
        operator!($trait $method $tch_method => bool => c16 => c16);
        operator!($trait $method $tch_method => c16 => bool => c16);

        /*
        ComplexFloat + ComplexFloat = ComplexFloat
        Uint8 + ComplexFloat = ComplexFloat
        ComplexFloat + Uint8 = ComplexFloat
        Int8 + ComplexFloat = ComplexFloat
        ComplexFloat + Int8 = ComplexFloat
        Int64 + ComplexFloat = ComplexFloat
        ComplexFloat + Int64 = ComplexFloat
        Int + ComplexFloat = ComplexFloat
        ComplexFloat + Int = ComplexFloat
        Int16 + ComplexFloat = ComplexFloat
        ComplexFloat + Int16 = ComplexFloat
        Float + ComplexHalf = ComplexFloat
        ComplexHalf + Float = ComplexFloat
        Float + ComplexFloat = ComplexFloat
        ComplexFloat + Float = ComplexFloat
        Bool + ComplexFloat = ComplexFloat
        ComplexFloat + Bool = ComplexFloat
        ComplexHalf + ComplexFloat = ComplexFloat
        ComplexFloat + ComplexHalf = ComplexFloat
        */
        operator!($trait $method $tch_method => c32 => c32 => c32);
        operator!($trait $method $tch_method => u8 => c32 => c32);
        operator!($trait $method $tch_method => c32 => u8 => c32);
        operator!($trait $method $tch_method => i8 => c32 => c32);
        operator!($trait $method $tch_method => c32 => i8 => c32);
        operator!($trait $method $tch_method => i64 => c32 => c32);
        operator!($trait $method $tch_method => c32 => i64 => c32);
        operator!($trait $method $tch_method => i32 => c32 => c32);
        operator!($trait $method $tch_method => c32 => i32 => c32);
        operator!($trait $method $tch_method => i16 => c32 => c32);
        operator!($trait $method $tch_method => c32 => i16 => c32);
        operator!($trait $method $tch_method => f32 => c16 => c32);
        operator!($trait $method $tch_method => c16 => f32 => c32);
        operator!($trait $method $tch_method => f32 => c32 => c32);
        operator!($trait $method $tch_method => c32 => f32 => c32);
        operator!($trait $method $tch_method => bool => c32 => c32);
        operator!($trait $method $tch_method => c32 => bool => c32);
        operator!($trait $method $tch_method => c16 => c32 => c32);
        operator!($trait $method $tch_method => c32 => c16 => c32);

        /*
        ComplexDouble + ComplexDouble = ComplexDouble
        Int64 + ComplexDouble = ComplexDouble
        ComplexDouble + Int64 = ComplexDouble
        Int + ComplexDouble = ComplexDouble
        ComplexDouble + Int = ComplexDouble
        Int16 + ComplexDouble = ComplexDouble
        ComplexDouble + Int16 = ComplexDouble
        Int8 + ComplexDouble = ComplexDouble
        ComplexDouble + Int8 = ComplexDouble
        Uint8 + ComplexDouble = ComplexDouble
        ComplexDouble + Uint8 = ComplexDouble
        Float + ComplexDouble = ComplexDouble
        ComplexDouble + Float = ComplexDouble
        Double + ComplexHalf = ComplexDouble
        ComplexHalf + Double = ComplexDouble
        Double + ComplexFloat = ComplexDouble
        ComplexFloat + Double = ComplexDouble
        Double + ComplexDouble = ComplexDouble
        ComplexDouble + Double = ComplexDouble
        ComplexHalf + ComplexDouble = ComplexDouble
        ComplexDouble + ComplexHalf = ComplexDouble
        ComplexFloat + ComplexDouble = ComplexDouble
        ComplexDouble + ComplexFloat = ComplexDouble
        Bool + ComplexDouble = ComplexDouble
        ComplexDouble + Bool = ComplexDouble
        */
        operator!($trait $method $tch_method => c64 => c64 => c64);
        operator!($trait $method $tch_method => i64 => c64 => c64);
        operator!($trait $method $tch_method => c64 => i64 => c64);
        operator!($trait $method $tch_method => i32 => c64 => c64);
        operator!($trait $method $tch_method => c64 => i32 => c64);
        operator!($trait $method $tch_method => i16 => c64 => c64);
        operator!($trait $method $tch_method => c64 => i16 => c64);
        operator!($trait $method $tch_method => i8 => c64 => c64);
        operator!($trait $method $tch_method => c64 => i8 => c64);
        operator!($trait $method $tch_method => u8 => c64 => c64);
        operator!($trait $method $tch_method => c64 => u8 => c64);
        operator!($trait $method $tch_method => f32 => c64 => c64);
        operator!($trait $method $tch_method => c64 => f32 => c64);
        operator!($trait $method $tch_method => f64 => c16 => c64);
        operator!($trait $method $tch_method => c16 => f64 => c64);
        operator!($trait $method $tch_method => f64 => c32 => c64);
        operator!($trait $method $tch_method => c32 => f64 => c64);
        operator!($trait $method $tch_method => f64 => c64 => c64);
        operator!($trait $method $tch_method => c64 => f64 => c64);
        operator!($trait $method $tch_method => c16 => c64 => c64);
        operator!($trait $method $tch_method => c64 => c16 => c64);
        operator!($trait $method $tch_method => c32 => c64 => c64);
        operator!($trait $method $tch_method => c64 => c32 => c64);
        operator!($trait $method $tch_method => bool => c64 => c64);
        operator!($trait $method $tch_method => c64 => bool => c64);
    };
}

def_op!(addmod Add add g_add);
def_op!(submod Sub sub g_sub);
def_op!(mulmod Mul mul g_mul);

macro_rules! def_div {
    ($($rhs:ty => $lhs:ty => $out:ty),* $(,)?) => {
        $(
            impl<S: Shape, D: Device> std::ops::Div<Tensor<S, D, $rhs>> for Tensor<S, D, $lhs> {
                type Output = Tensor<S, D, $out>;
                fn div(self, rhs: Tensor<S, D, $rhs>) -> Self::Output {
                    Tensor {
                        repr: self.repr.g_div(&rhs.repr),
                        ..Default::default()
                    }
                }
            }
            impl<S: Shape, D: Device> std::ops::Div<&Tensor<S, D, $rhs>> for Tensor<S, D, $lhs> {
                type Output = Tensor<S, D, $out>;
                fn div(self, rhs: &Tensor<S, D, $rhs>) -> Self::Output {
                    Tensor {
                        repr: self.repr.g_div(&rhs.repr),
                        ..Default::default()
                    }
                }
            }
            impl<S: Shape, D: Device> std::ops::Div<&Tensor<S, D, $rhs>> for &Tensor<S, D, $lhs> {
                type Output = Tensor<S, D, $out>;

                fn div(self, rhs: &Tensor<S, D, $rhs>) -> Self::Output {
                    Tensor {
                        repr: self.repr.g_div(&rhs.repr),
                        ..Default::default()
                    }
                }
            }
            impl<S: Shape, D: Device> std::ops::Div<Tensor<S, D, $rhs>> for &Tensor<S, D, $lhs> {
                type Output = Tensor<S, D, $out>;

                fn div(self, rhs: Tensor<S, D, $rhs>) -> Self::Output {
                    Tensor {
                        repr: self.repr.g_div(&rhs.repr),
                        ..Default::default()
                    }
                }
            }
        )*
    };
}

/*
Bool / Uint8 = Float
Bool / Int8 = Float
Bool / Int16 = Float
Bool / Int = Float
Bool / Int64 = Float
Bool / Half = Half
Bool / Float = Float
Bool / Double = Double
Bool / Bool = Float

Uint8 / Int8 = Float
Uint8 / Bool = Float
Uint8 / Int16 = Float
Uint8 / Int = Float
Uint8 / Int64 = Float
Uint8 / Half = Half
Uint8 / Float = Float
Uint8 / Double = Double
Uint8 / ComplexFloat = ComplexFloat
Uint8 / ComplexDouble = ComplexDouble
Uint8 / Uint8 = Float

Int8 / Uint8 = Float
Int8 / Bool = Float
Int8 / Int16 = Float
Int8 / Int = Float
Int8 / Int64 = Float
Int8 / Half = Half
Int8 / Float = Float
Int8 / Double = Double
Int8 / ComplexFloat = ComplexFloat
Int8 / ComplexDouble = ComplexDouble
Int8 / Int8 = Float

Int16 / Uint8 = Float
Int16 / Bool = Float
Int16 / Int8 = Float
Int16 / Int = Float
Int16 / Int64 = Float
Int16 / Half = Half
Int16 / Float = Float
Int16 / Double = Double
Int16 / ComplexFloat = ComplexFloat
Int16 / ComplexDouble = ComplexDouble
Int16 / Int16 = Float

Int / Uint8 = Float
Int / Bool = Float
Int / Int8 = Float
Int / Int16 = Float
Int / Int64 = Float
Int / Half = Half
Int / Float = Float
Int / Double = Double
Int / ComplexFloat = ComplexFloat
Int / ComplexDouble = ComplexDouble
Int / Int = Float

Int64 / Uint8 = Float
Int64 / Bool = Float
Int64 / Int8 = Float
Int64 / Int16 = Float
Int64 / Int = Float
Int64 / Half = Half
Int64 / Float = Float
Int64 / Double = Double
Int64 / ComplexFloat = ComplexFloat
Int64 / ComplexDouble = ComplexDouble
Int64 / Int64 = Float

Half / Int = Half
Half / Bool = Half
Half / Float = Float
Half / Double = Double
Half / Uint8 = Half
Half / Int8 = Half
Half / Int16 = Half
Half / ComplexFloat = ComplexFloat
Half / ComplexDouble = ComplexDouble
Half / Int64 = Half
Half / Half = Half

Float / Int64 = Float
Float / Bool = Float
Float / Half = Float
Float / Double = Double
Float / ComplexFloat = ComplexFloat
Float / ComplexDouble = ComplexDouble
Float / Int = Float
Float / Uint8 = Float
Float / Int8 = Float
Float / Int16 = Float
Float / Float = Float

Double / Int = Double
Double / Bool = Double
Double / Int64 = Double
Double / Half = Double
Double / Float = Double
Double / Uint8 = Double
Double / Int8 = Double
Double / Int16 = Double
Double / ComplexFloat = ComplexDouble
Double / ComplexDouble = ComplexDouble
Double / Double = Double

ComplexFloat / Half = ComplexFloat
ComplexFloat / Bool = ComplexFloat
ComplexFloat / Float = ComplexFloat
ComplexFloat / Double = ComplexDouble
ComplexFloat / ComplexDouble = ComplexDouble
ComplexFloat / Int64 = ComplexFloat
ComplexFloat / Int = ComplexFloat
ComplexFloat / Int8 = ComplexFloat
ComplexFloat / Int16 = ComplexFloat
ComplexFloat / Uint8 = ComplexFloat
ComplexFloat / ComplexFloat = ComplexFloat

ComplexDouble / Uint8 = ComplexDouble
ComplexDouble / Bool = ComplexDouble
ComplexDouble / Int8 = ComplexDouble
ComplexDouble / Int16 = ComplexDouble
ComplexDouble / Int = ComplexDouble
ComplexDouble / Int64 = ComplexDouble
ComplexDouble / Half = ComplexDouble
ComplexDouble / Float = ComplexDouble
ComplexDouble / Double = ComplexDouble
ComplexDouble / ComplexFloat = ComplexDouble
ComplexDouble / ComplexDouble = ComplexDouble
*/
def_div! {
    u8 => u8 => f32,
    u8 => i8 => f32,
    u8 => i16 => f32,
    u8 => i32 => f32,
    u8 => i64 => f32,
    u8 => f32 => f32,
    u8 => f64 => f64,
    u8 => c32 => c32,
    u8 => c64 => c64,
    u8 => bool => f32,

    i8 => u8 => f32,
    i8 => i8 => f32,
    i8 => i16 => f32,
    i8 => i32 => f32,
    i8 => i64 => f32,
    i8 => f32 => f32,
    i8 => f64 => f64,
    i8 => c32 => c32,
    i8 => c64 => c64,
    i8 => bool => f32,

    i16 => u8 => f32,
    i16 => i8 => f32,
    i16 => i16 => f32,
    i16 => i32 => f32,
    i16 => i64 => f32,
    i16 => f32 => f32,
    i16 => f64 => f64,
    i16 => c32 => c32,
    i16 => c64 => c64,
    i16 => bool => f32,

    i32 => u8 => f32,
    i32 => i8 => f32,
    i32 => i16 => f32,
    i32 => i32 => f32,
    i32 => i64 => f32,
    i32 => f32 => f32,
    i32 => f64 => f64,
    i32 => c32 => c32,
    i32 => c64 => c64,
    i32 => bool => f32,

    i64 => u8 => f32,
    i64 => i8 => f32,
    i64 => i16 => f32,
    i64 => i32 => f32,
    i64 => i64 => f32,
    i64 => f32 => f32,
    i64 => f64 => f64,
    i64 => c32 => c32,
    i64 => c64 => c64,
    i64 => bool => f32,

    f32 => i32 => f32,
    f32 => i64 => f32,
    f32 => f32 => f32,
    f32 => f64 => f64,
    f32 => c16 => c32,
    f32 => c32 => c32,
    f32 => c64 => c64,
    f32 => bool => f32,
    f32 => u8 => f32,
    f32 => i8 => f32,
    f32 => i16 => f32,

    f64 => i32 => f64,
    f64 => i64 => f64,
    f64 => f32 => f64,
    f64 => f64 => f64,
    f64 => c16 => c64,
    f64 => c32 => c64,
    f64 => c64 => c64,
    f64 => bool => f64,
    f64 => u8 => f64,
    f64 => i8 => f64,
    f64 => i16 => f64,

    c16 => f32 => c32,
    c16 => bool => c16,
    c16 => f64 => c64,
    c16 => c16 => c16,
    c16 => c32 => c32,
    c16 => c64 => c64,
    c16 => u8 => c16,
    c16 => i8 => c16,
    c16 => i16 => c16,
    c16 => i32 => c16,

    c32 => f32 => c32,
    c32 => bool => c32,
    c32 => f64 => c64,
    c32 => c16 => c32,
    c32 => c32 => c32,
    c32 => c64 => c64,
    c32 => u8 => c32,
    c32 => i8 => c32,
    c32 => i16 => c32,
    c32 => i32 => c32,
}
#[cfg(feature = "half")]
use crate::kind::f16;
#[cfg(feature = "half")]
def_div! {
    u8 => f16 => f16,
    i8 => f16 => f16,
    i16 => f16 => f16,
    i32 => f16 => f16,
    i64 => f16 => f16,
    f16 => f16 => f16,
    f32 => f16 => f32,
    f64 => f16 => f64,
    c16 => f16 => c16,
    c32 => f16 => c32,
    c64 => f16 => c64,
    bool => f16 => f16,

    f16 => u8 => f16,
    f16 => i8 => f16,
    f16 => i16 => f16,
    f16 => i32 => f16,
    f16 => i64 => f16,
    f16 => f32 => f16,
    f16 => f64 => f16,
    f16 => c16 => c16,
    f16 => c32 => c32,
    f16 => c64 => c64,
    f16 => bool => f16,
}

macro_rules! def_assign {
    ($trait:ident $method:ident $tch_method:ident $($type:ty)+) => {
        $(
            impl<S: Shape, D: Device> std::ops::$trait<Tensor<S, D, $type>> for Tensor<S, D, $type> {
                fn $method(&mut self, rhs: Tensor<S, D, $type>) {
                    let _ = self.repr.$tch_method(&rhs.repr);
                }
            }
            impl<S: Shape, D: Device> std::ops::$trait<&Tensor<S, D, $type>> for Tensor<S, D, $type> {
                fn $method(&mut self, rhs: &Tensor<S, D, $type>) {
                    let _ = self.repr.$tch_method(&rhs.repr);
                }
            }
        )+
    };
}

def_assign!(AddAssign add_assign g_add_ u8 i8 i16 i32 i64 f32 f64 c16 c32 c64 bool);
def_assign!(SubAssign sub_assign g_sub_ u8 i8 i16 i32 i64 f32 f64 c16 c32 c64 bool);
def_assign!(MulAssign mul_assign g_mul_ u8 i8 i16 i32 i64 f32 f64 c16 c32 c64 bool);
def_assign!(DivAssign div_assign g_div_ u8 i8 i16 i32 i64 f32 f64 c16 c32 c64 bool);

#[cfg(feature = "half")]
mod assign_half {
    use super::*;
    def_assign!(AddAssign add_assign g_add_ f16);
    def_assign!(SubAssign sub_assign g_sub_ f16);
    def_assign!(MulAssign mul_assign g_mul_ f16);
    def_assign!(DivAssign div_assign g_div_ f16);
}
