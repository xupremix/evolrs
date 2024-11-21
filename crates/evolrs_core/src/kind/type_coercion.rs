#[cfg(feature = "half")]
use super::f16;

use super::{c16, c32, c64, Kind};

pub trait Same<T> {}
impl<T> Same<T> for T {}

pub trait Coerce<Base: Kind>: Kind {
    type To: Kind;
}
pub trait DivCoerce<Base: Kind>: Kind {
    type To: Kind;
}

macro_rules! def_coercion {
    ($($to:ty: { $($lhs:ty, $rhs:ty);* $(;)? }),* $(,)?) => {
        $(
            $(
                impl Coerce<$rhs> for $lhs {
                    type To = $to;
                }
            )*
        )*
    };
    (@div $($to:ty: { $($lhs:ty, $rhs:ty);* $(;)? }),* $(,)?) => {
        $(
            $(
                impl DivCoerce<$rhs> for $lhs {
                    type To = $to;
                }
            )*
        )*
    };
}

def_coercion! {
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

#[cfg(feature = "half")]
def_coercion! {
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

def_coercion! {
    @div
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
def_coercion! {
    @div
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

#[cfg(test)]
mod tests {
    use super::*;
    fn same<U, T: Same<U>>() {}

    macro_rules! def_test {
        ($($name:ident $to:ty: { $($lhs:ty, $rhs:ty);* $(;)? }),* $(,)?) => {
            $(
                #[test]
                fn $name() {
                    $(
                        same::<$to, <$lhs as Coerce<$rhs>>::To>();
                    )*
                }
            )*
        };
        (@div $($name:ident $to:ty: { $($lhs:ty, $rhs:ty);* $(;)? }),* $(,)?) => {
            $(
                #[test]
                fn $name() {
                    $(
                        same::<$to, <$lhs as DivCoerce<$rhs>>::To>();
                    )*
                }
            )*
        };
    }

    def_test! {
        test_bool
        bool: {
            bool, bool;
        },
        test_u8
        u8: {
            u8, u8;
            bool, u8;
            u8, bool;
        },
        test_i8
        i8: {
            i8, i8;
            bool, i8;
            i8, bool;
        },
        test_i16
        i16: {
            u8, i16;
            i8, i16;
            i16, u8;
            i16, i8;
            i16, i16;
            bool, i16;
            i16, bool;
        },
        test_i32
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
        test_i64
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
        test_f32
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
        test_f64
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
        test_c16
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
        test_c32
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
        test_c64
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

    #[cfg(feature = "half")]
    def_test! {
        test_f16
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
        test_f16_f32
        f32: {
            f16, f32;
            f32, f16;
        },
        test_f16_f64
        f64: {
            f16, f64;
            f64, f16;
        },
        test_f16_c16
        c16: {
            f16, c16;
            c16, f16;
        },
        test_f16_c32
        c32: {
            f16, c32;
            c32, f16;
        },
        test_f16_c64
        c64: {
            f16, c64;
            c64, f16;
        },
    }

    def_test! {
        @div
        div_test_f32
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
        div_test_f64
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
        div_test_c32
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
        div_test_c64
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
    def_test! {
        @div
        div_test_f16
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
        div_test_f16_f32
        f32: {
            f16, f32;
            f32, f16;
        },
        div_test_f16_f64
        f64: {
            f16, f64;
            f64, f16;
        },
        div_test_f16_c32
        c32: {
            c32, f16;
            f16, c32;
        },
        div_test_f16_c64
        c64: {
            f16, c64;
            c64, f16;
        },
    }
}
