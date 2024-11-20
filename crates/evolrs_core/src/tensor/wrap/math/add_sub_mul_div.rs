use crate::device::Device;
use crate::kind::Kind;
use crate::shapes::shape::Shape;
use crate::tensor::Tensor;

#[cfg(feature = "broadcast-semantics")]
use crate::shapes::broadcast::{Broadcast, BroadcastInplace};

macro_rules! def_fn {
    ($trait:ident $fn:ident $tch_fn:ident $fn_:ident $tch_fn_:ident $( $ris:ty: { $( $x:ty, $y:ty );* $(;)? } ),* $(,)?) => {
        pub trait $trait<S: Shape, D: Device, RHS: Kind, RIS: Kind> {
            #[cfg(not(feature = "broadcast-semantics"))]
            fn $fn(&self, rhs: &Tensor<S, D, RHS>) -> Tensor<S, D, RIS>;

            #[cfg(feature = "broadcast-semantics")]
            fn $fn<Dst: Shape, Rhs: Broadcast<S, Dst>>(&self, rhs: &Tensor<Rhs, D, RHS>) -> Tensor<Dst, D, RIS>;
        }

        impl<S: Shape, D: Device, K: Kind> Tensor<S, D, K> {
            #[cfg(not(feature = "broadcast-semantics"))]
            pub fn $fn_(&mut self, rhs: &Tensor<S, D, K>) -> Tensor<S, D, K> {
                Tensor {
                    repr: self.repr.$tch_fn_(&rhs.repr),
                    ..Default::default()
                }
            }

            #[cfg(feature = "broadcast-semantics")]
            pub fn $fn_<Rhs: BroadcastInplace<S>>(&mut self, rhs: &Tensor<Rhs, D, K>) -> Tensor<S, D, K> {
                #![allow(path_statements)]
                Rhs::BROADCAST_INPLACE_CHECK;
                Tensor {
                    repr: self.repr.$tch_fn_(&rhs.repr),
                    ..Default::default()
                }
            }
        }

        $(
            $(
                impl<S: Shape, D: Device> $trait<S, D, $y, $ris> for Tensor<S, D, $x> {
                    #[cfg(not(feature = "broadcast-semantics"))]
                    fn $fn(&self, rhs: &Tensor<S, D, $y>) -> Tensor<S, D, $ris> {
                        Tensor {
                            repr: self.repr.$tch_fn(&rhs.repr),
                            ..Default::default()
                        }
                    }

                    #[cfg(feature = "broadcast-semantics")]
                    fn $fn<Dst: Shape, Rhs: Broadcast<S, Dst>>(&self, rhs: &Tensor<Rhs, D, $y>) -> Tensor<Dst, D, $ris> {
                        #![allow(path_statements)]
                        Rhs::BROADCAST_CHECK;
                        Tensor {
                            repr: self.repr.$tch_fn(&rhs.repr),
                            ..Default::default()
                        }
                    }
                }
            )*
        )*
    };
    (@no_def $trait:ident $fn:ident $tch_fn:ident $fn_:ident $tch_fn_:ident $( $ris:ty: { $( $x:ty, $y:ty );* $(;)? } ),* $(,)?) => {
        $(
            $(
                impl<S: Shape, D: Device> $trait<S, D, $y, $ris> for Tensor<S, D, $x> {
                    #[cfg(not(feature = "broadcast-semantics"))]
                    fn $fn(&self, rhs: &Tensor<S, D, $y>) -> Tensor<S, D, $ris> {
                        Tensor {
                            repr: self.repr.$tch_fn(&rhs.repr),
                            ..Default::default()
                        }
                    }

                    #[cfg(feature = "broadcast-semantics")]
                    fn $fn<Dst: Shape, Rhs: Broadcast<S, Dst>>(&self, rhs: &Tensor<Rhs, D, $y>) -> Tensor<Dst, D, $ris> {
                        #![allow(path_statements)]
                        Rhs::BROADCAST_CHECK;
                        Tensor {
                            repr: self.repr.$tch_fn(&rhs.repr),
                            ..Default::default()
                        }
                    }
                }
            )*
        )*
    };
}

use crate::kind::{c16, c32, c64};

macro_rules! perm {
    ($( $trait:ident $fn:ident $tch_fn:ident $fn_:ident $tch_fn_:ident ),* $(,)?) => {
        $(
            def_fn! {
                $trait $fn $tch_fn $fn_ $tch_fn_
                bool: {
                    bool, bool;
                },
                u8: {
                    u8, u8;
                    bool, u8;
                    u8, bool;
                },
                i8: {
                    i8, i8;
                    bool, i8;
                    i8, bool;
                },
                i16: {
                    u8, i16;
                    i8, i16;
                    i16, u8;
                    i16, i8;
                    i16, i16;
                    bool, i16;
                    i16, bool;
                },
                i32: {
                    i32, i32;
                    u8, i32;
                    i32, u8;
                    i8, i32;
                    i32, i8;
                    i16, i32;
                    i32, i16;
                    bool, i32;
                    i32, bool;
                },
                i64: {
                    i64, i64;
                    u8, i64;
                    i64, u8;
                    i8, i64;
                    i64, i8;
                    i16, i64;
                    i64, i16;
                    i32, i64;
                    i64, i32;
                    bool, i64;
                    i64, bool;
                },
                f32: {
                    f32, f32;
                    u8, f32;
                    f32, u8;
                    i8, f32;
                    f32, i8;
                    i16, f32;
                    f32, i16;
                    i32, f32;
                    f32, i32;
                    i64, f32;
                    f32, i64;
                    bool, f32;
                    f32, bool;
                },
                f64: {
                    f64, f64;
                    i8, f64;
                    f64, i8;
                    u8, f64;
                    f64, u8;
                    i16, f64;
                    f64, i16;
                    i32, f64;
                    f64, i32;
                    i64, f64;
                    f64, i64;
                    f32, f64;
                    f64, f32;
                    bool, f64;
                    f64, bool;
                },
                c16: {
                    c16, c16;
                    u8, c16;
                    c16, u8;
                    i8, c16;
                    c16, i8;
                    i16, c16;
                    c16, i16;
                    i32, c16;
                    c16, i32;
                    i64, c16;
                    c16, i64;
                    bool, c16;
                    c16, bool;
                },
                c32: {
                    c32, c32;
                    u8, c32;
                    c32, u8;
                    i8, c32;
                    c32, i8;
                    i16, c32;
                    c32, i16;
                    i32, c32;
                    c32, i32;
                    i64, c32;
                    c32, i64;
                    f32, c16;
                    c16, f32;
                    f32, c32;
                    c32, f32;
                    bool, c32;
                    c32, bool;
                    c16, c32;
                    c32, c16;
                },
                c64: {
                    c64, c64;
                    i64, c64;
                    c64, i64;
                    i32, c64;
                    c64, i32;
                    i16, c64;
                    c64, i16;
                    i8, c64;
                    c64, i8;
                    u8, c64;
                    c64, u8;
                    f32, c64;
                    c64, f32;
                    f64, c16;
                    c16, f64;
                    f64, c32;
                    c32, f64;
                    f64, c64;
                    c64, f64;
                    c16, c64;
                    c64, c16;
                    c32, c64;
                    c64, c32;
                    bool, c64;
                    c64, bool;
                },
            }
        )*
    };
    (@no_def $( $trait:ident $fn:ident $tch_fn:ident $fn_:ident $tch_fn_:ident ),* $(,)?) => {
        $(
            def_fn! {
                @no_def
                $trait $fn $tch_fn $fn_ $tch_fn_
                f16: {
                    f16, f16;
                    u8, f16;
                    f16, u8;
                    i8, f16;
                    f16, i8;
                    i16, f16;
                    f16, i16;
                    i32, f16;
                    f16, i32;
                    i64, f16;
                    f16, i64;
                    bool, f16;
                    f16, bool;
                },
                f32: {
                    f16, f32;
                    f32, f16;
                },
                f64: {
                    f16, f64;
                    f64, f16;
                },
                c16: {
                    f16, c16;
                    c16, f16;
                },
                c32: {
                    f16, c32;
                    c32, f16;
                },
                c64: {
                    f16, c64;
                    c64, f16;
                },
            }
        )*
    };
}

perm! {
    Add add g_add add_ g_add_,
    Sub sub g_sub sub_ g_sub_,
    Mul mul g_mul mul_ g_mul_,
}

#[cfg(feature = "half")]
use crate::kind::f16;

#[cfg(feature = "half")]
perm! {
    @no_def
    Add add g_add add_ g_add_,
    Sub sub g_sub sub_ g_sub_,
    Mul mul g_mul mul_ g_mul_,
}

def_fn! {
    Div div g_div div_ g_div_
    f32: {
        bool, u8;
        bool, i8;
        bool, i16;
        bool, i32;
        bool, i64;
        bool, f32;
        bool, bool;
        u8, i8;
        u8, bool;
        u8, i16;
        u8, i32;
        u8, i64;
        u8, f32;
        u8, u8;
        i8, u8;
        i8, bool;
        i8, i16;
        i8, i32;
        i8, i64;
        i8, f32;
        i8, i8;
        i64, u8;
        i64, bool;
        i64, i8;
        i64, i16;
        i64, i32;
        i64, f32;
        i64, i64;
        f32, i64;
        f32, bool;
        f32, i32;
        f32, u8;
        f32, i8;
        f32, i16;
        f32, f32;
        i16, u8;
        i16, bool;
        i16, i8;
        i16, i32;
        i16, i64;
        i16, f32;
        i16, i16;
        i32, u8;
        i32, bool;
        i32, i8;
        i32, i16;
        i32, i64;
        i32, f32;
        i32, i32;
    },
    f64: {
        i8, f64;
        bool, f64;
        u8, f64;
        i16, f64;
        i32, f64;
        i64, f64;
        f32, f64;
        f64, i32;
        f64, bool;
        f64, i64;
        f64, f32;
        f64, u8;
        f64, i8;
        f64, i16;
        f64, f64;
    },
    c32: {
        c32, bool;
        c32, f32;
        c32, i64;
        c32, i32;
        c32, i8;
        c32, i16;
        c32, u8;
        c32, c32;
        f32, c32;
        i64, c32;
        i32, c32;
        i16, c32;
        i8, c32;
        u8, c32;
    },
    c64: {
        u8, c64;
        i8, c64;
        i16, c64;
        i32, c64;
        i64, c64;
        f32, c64;
        f64, c32;
        f64, c64;
        c32, c64;
        c32, f64;
        c64, u8;
        c64, bool;
        c64, i8;
        c64, i16;
        c64, i32;
        c64, i64;
        c64, f32;
        c64, f64;
        c64, c32;
        c64, c64;
    },
}

#[cfg(feature = "half")]
def_fn! {
    @no_def
    Div div g_div div_ g_div_
    f16: {
        bool, f16;
        u8, f16;
        i16, f16;
        i32, f16;
        i64, f16;
        f16, i32;
        f16, bool;
        f16, u8;
        f16, i8;
        f16, i16;
        f16, i64;
        f16, f16;
        i8, f16;
    },
    f32: {
       f16, f32;
       f32, f16;
    },
    f64: {
       f16, f64;
       f64, f16;
    },
    c32: {
      c32, f16;
      f16, c32;
    },
    c64: {
      f16, c64;
      c64, f16;
    },
}
