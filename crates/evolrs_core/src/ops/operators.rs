use crate::{
    device::Device,
    kind::{
        restriction::composite::{IntOrBool, NotBool},
        type_coercion::{Coerce, DivCoerce, Same},
        Kind,
    },
    shapes::shape::Shape,
    tensor::Tensor,
};

macro_rules! op {
    ($( $kind:ident $coerce:ident $trait:ident $trait_:ident $fn:ident $tch_fn:ident $fn_:ident $tch_fn_:ident $($restr:ident)? ),* $(,)?) => {
        $(
            impl<S: Shape, D: Device, K: $kind, K2: $coerce<K> $(+ $restr)?> std::ops::$trait<Tensor<S, D, K2>>
                for Tensor<S, D, K>
            {
                type Output = Tensor<S, D, K2::To>;
                fn $fn(self, rhs: Tensor<S, D, K2>) -> Self::Output {
                    Tensor {
                        repr: self.repr.$tch_fn(&rhs.repr),
                        ..Default::default()
                    }
                }
            }
            impl<S: Shape, D: Device, K: $kind, K2: $coerce<K> $(+ $restr)?> std::ops::$trait<&Tensor<S, D, K2>>
                for Tensor<S, D, K>
            {
                type Output = Tensor<S, D, K2::To>;
                fn $fn(self, rhs: &Tensor<S, D, K2>) -> Self::Output {
                    Tensor {
                        repr: self.repr.$tch_fn(&rhs.repr),
                        ..Default::default()
                    }
                }
            }
            impl<S: Shape, D: Device, K: $kind, K2: $coerce<K> $(+ $restr)?> std::ops::$trait<Tensor<S, D, K2>>
                for &Tensor<S, D, K>
            {
                type Output = Tensor<S, D, K2::To>;
                fn $fn(self, rhs: Tensor<S, D, K2>) -> Self::Output {
                    Tensor {
                        repr: self.repr.$tch_fn(&rhs.repr),
                        ..Default::default()
                    }
                }
            }
            impl<S: Shape, D: Device, K: $kind, K2: $coerce<K> $(+ $restr)?> std::ops::$trait<&Tensor<S, D, K2>>
                for &Tensor<S, D, K>
            {
                type Output = Tensor<S, D, K2::To>;
                fn $fn(self, rhs: &Tensor<S, D, K2>) -> Self::Output {
                    Tensor {
                        repr: self.repr.$tch_fn(&rhs.repr),
                        ..Default::default()
                    }
                }
            }
            impl<S: Shape, D: Device, K: $kind, K2: $coerce<K> $(+ $restr)?> std::ops::$trait_<Tensor<S, D, K2>> for Tensor<S, D, K>
            where
                K: Same<K2::To>
            {
                fn $fn_(&mut self, rhs: Tensor<S, D, K2>) {
                    let _ = self.repr.$tch_fn_(&rhs.repr);
                }
            }
            impl<S: Shape, D: Device, K: $kind, K2: $coerce<K> $(+ $restr)?> std::ops::$trait_<&Tensor<S, D, K2>> for Tensor<S, D, K>
            where
                K: Same<K2::To>
            {
                fn $fn_(&mut self, rhs: &Tensor<S, D, K2>) {
                    let _ = self.repr.$tch_fn_(&rhs.repr);
                }
            }
        )*
    };
}

op! {
   Kind Coerce Add AddAssign add g_add add_assign g_add_,
   Kind Coerce Sub SubAssign sub g_sub sub_assign g_sub_,
   Kind Coerce Mul MulAssign mul g_mul mul_assign g_mul_,
   Kind DivCoerce Div DivAssign div g_div div_assign g_div_,
   IntOrBool Coerce BitAnd BitAndAssign bitand bitwise_and_tensor bitand_assign bitwise_and_tensor_ IntOrBool,
   IntOrBool Coerce BitOr BitOrAssign bitor bitwise_or_tensor bitor_assign bitwise_or_tensor_ IntOrBool,
   IntOrBool Coerce BitXor BitXorAssign bitxor bitwise_xor_tensor bitxor_assign bitwise_xor_tensor_ IntOrBool,
   IntOrBool Coerce Shl ShlAssign shl bitwise_left_shift shl_assign bitwise_left_shift_ IntOrBool,
   IntOrBool Coerce Shr ShrAssign shr bitwise_right_shift shr_assign bitwise_right_shift_ IntOrBool,
}

impl<S: Shape, D: Device, K: IntOrBool> std::ops::Not for Tensor<S, D, K> {
    type Output = Self;
    fn not(self) -> Self::Output {
        Tensor {
            repr: self.repr.bitwise_not(),
            ..Default::default()
        }
    }
}

impl<S: Shape, D: Device, K: IntOrBool> std::ops::Not for &Tensor<S, D, K> {
    type Output = Tensor<S, D, K>;
    fn not(self) -> Self::Output {
        Tensor {
            repr: self.repr.bitwise_not(),
            ..Default::default()
        }
    }
}

impl<S: Shape, D: Device, K: NotBool> std::ops::Neg for Tensor<S, D, K> {
    type Output = Tensor<S, D, K>;
    fn neg(self) -> Self::Output {
        Tensor {
            repr: -self.repr,
            ..Default::default()
        }
    }
}

impl<S: Shape, D: Device, K: NotBool> std::ops::Neg for &Tensor<S, D, K> {
    type Output = Tensor<S, D, K>;
    fn neg(self) -> Self::Output {
        Tensor {
            repr: -&self.repr,
            ..Default::default()
        }
    }
}

impl<S: Shape, D: Device, K: Kind> Clone for Tensor<S, D, K> {
    fn clone(&self) -> Self {
        Self {
            repr: self.repr.copy(),
            ..Default::default()
        }
    }
}

impl<S: Shape, D: Device, K: Kind, K2: Kind> PartialEq<Tensor<S, D, K2>> for Tensor<S, D, K> {
    fn eq(&self, other: &Tensor<S, D, K2>) -> bool {
        self.repr == other.repr
    }
}

// TODO: add tests for new impls
#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Cpu;
    use crate::shapes::shape::Rank2;
    use crate::tensor::ToTchTensor;

    macro_rules! def_test {
        ($name:ident => $op:tt => $expected:expr => $type:ty => $change:ty) => {
            #[test]
            fn $name() {
                let t1: Tensor<Rank2<2, 3>, Cpu, $type> = Tensor::ones();
                let t2: Tensor<Rank2<2, 3>, Cpu, $type> = Tensor::ones();
                let t3: Tensor<Rank2<2, 3>, Cpu, $change> = t1 $op t2;
                assert_eq!(t3.to_tch().sum(None), $expected);

                let t1: Tensor<Rank2<2, 3>, Cpu, $type> = Tensor::ones();
                let t2: Tensor<Rank2<2, 3>, Cpu, $type> = Tensor::ones();
                let t3: Tensor<Rank2<2, 3>, Cpu, $change> = &t1 $op t2;
                assert_eq!(t3.to_tch().sum(None), $expected);

                let t1: Tensor<Rank2<2, 3>, Cpu, $type> = Tensor::ones();
                let t2: Tensor<Rank2<2, 3>, Cpu, $type> = Tensor::ones();
                let t3: Tensor<Rank2<2, 3>, Cpu, $change> = t1 $op &t2;
                assert_eq!(t3.to_tch().sum(None), $expected);

                let t1: Tensor<Rank2<2, 3>, Cpu, $type> = Tensor::ones();
                let t2: Tensor<Rank2<2, 3>, Cpu, $type> = Tensor::ones();
                let t3: Tensor<Rank2<2, 3>, Cpu, $change> = &t1 $op &t2;
                assert_eq!(t3.to_tch().sum(None), $expected);
            }
        };
        (@assign $name:ident => $op:tt => $expected:expr) => {
            #[test]
            fn $name() {
                let mut t1: Tensor<crate::shapes::shape::Rank2<2, 3>> = Tensor::ones();
                let t2: Tensor<crate::shapes::shape::Rank2<2, 3>> = Tensor::ones();
                t1 $op t2;
                assert_eq!(t1.to_tch().sum(None), $expected);
            }
        };
    }

    def_test!(test_add_i32 => + => tch::Tensor::from(12) => i32 => i32);
    def_test!(test_add_i64 => + => tch::Tensor::from(12) => i64 => i64);
    def_test!(test_add_f32 => + => tch::Tensor::from(12.0) => f32 => f32);
    def_test!(test_add_f64 => + => tch::Tensor::from(12.0) => f64 => f64);

    def_test!(test_sub_i32 => - => tch::Tensor::from(0) => i32 => i32);
    def_test!(test_sub_i64 => - => tch::Tensor::from(0) => i64 => i64);
    def_test!(test_sub_f32 => - => tch::Tensor::from(0.0) => f32 => f32);
    def_test!(test_sub_f64 => - => tch::Tensor::from(0.0) => f64 => f64);

    def_test!(test_mul_i32 => * => tch::Tensor::from(6) => i32 => i32);
    def_test!(test_mul_i64 => * => tch::Tensor::from(6) => i64 => i64);
    def_test!(test_mul_f32 => * => tch::Tensor::from(6.0) => f32 => f32);
    def_test!(test_mul_f64 => * => tch::Tensor::from(6.0) => f64 => f64);

    def_test!(test_div_i32 => / => tch::Tensor::from(6.0) => i32 => f32);
    def_test!(test_div_i64 => / => tch::Tensor::from(6.0) => i64 => f32);
    def_test!(test_div_f32 => / => tch::Tensor::from(6.0) => f32 => f32);
    def_test!(test_div_f64 => / => tch::Tensor::from(6.0) => f64 => f64);

    def_test!(@assign test_add_assign_i32 => += => tch::Tensor::from(12));
    def_test!(@assign test_add_assign_i64 => += => tch::Tensor::from(12));
    def_test!(@assign test_add_assign_f32 => += => tch::Tensor::from(12.0));
    def_test!(@assign test_add_assign_f64 => += => tch::Tensor::from(12.0));

    def_test!(@assign test_sub_assign_i32 => -= => tch::Tensor::from(0));
    def_test!(@assign test_sub_assign_i64 => -= => tch::Tensor::from(0));
    def_test!(@assign test_sub_assign_f32 => -= => tch::Tensor::from(0.0));
    def_test!(@assign test_sub_assign_f64 => -= => tch::Tensor::from(0.0));

    def_test!(@assign test_mul_assign_i32 => *= => tch::Tensor::from(6));
    def_test!(@assign test_mul_assign_i64 => *= => tch::Tensor::from(6));
    def_test!(@assign test_mul_assign_f32 => *= => tch::Tensor::from(6.0));
    def_test!(@assign test_mul_assign_f64 => *= => tch::Tensor::from(6.0));

    def_test!(@assign test_div_assign_f32 => /= => tch::Tensor::from(6.0));
    def_test!(@assign test_div_assign_f64 => /= => tch::Tensor::from(6.0));
}
