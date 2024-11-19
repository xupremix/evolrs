use crate::{device::Device, shapes::shape::Shape, tensor::Tensor};

macro_rules! operator {
    ($trait:ident $assign_trait:ident $method:ident $assign_method:ident $tch_method:ident $tch_tensor_method:ident $tch_assign_tensor_method:ident $type:ty $( => $base:ty )?) => {
        impl<S: Shape, D: Device> std::ops::$trait<Tensor<S, D, $type>> for $type {
            type Output = Tensor<S, D, $type>;
            fn $method(self, rhs: Tensor<S, D, $type>) -> Self::Output {
                Tensor {
                    repr: self.$tch_method(&rhs.repr),
                    ..Default::default()
                }
            }
        }
        impl<S: Shape, D: Device> std::ops::$trait<&Tensor<S, D, $type>> for $type {
            type Output = Tensor<S, D, $type>;

            fn $method(self, rhs: &Tensor<S, D, $type>) -> Self::Output {
                Tensor {
                    repr: self.$tch_method(&rhs.repr),
                    ..Default::default()
                }
            }
        }
        impl<S: Shape, D: Device> std::ops::$trait<$type> for Tensor<S, D, $type> {
            type Output = Tensor<S, D, $type>;

            fn $method(self, rhs: $type) -> Self::Output {
                Tensor {
                    repr: self.repr.$tch_tensor_method(rhs $( as $base )? ),
                    ..Default::default()
                }
            }
        }
        impl<S: Shape, D: Device> std::ops::$trait<$type> for &Tensor<S, D, $type> {
            type Output = Tensor<S, D, $type>;

            fn $method(self, rhs: $type) -> Self::Output {
                Tensor {
                    repr: self.repr.$tch_tensor_method(rhs $( as $base )?),
                    ..Default::default()
                }
            }
        }
        impl<S: Shape, D: Device> std::ops::$assign_trait<$type> for Tensor<S, D, $type> {
            fn $assign_method(&mut self, rhs: $type) {
                let _ = self.repr.$tch_assign_tensor_method(rhs $( as $base )?);
            }
        }
    };
    (@change $trait:ident $assign_trait:ident $method:ident $assign_method:ident $tch_method:ident $tch_tensor_method:ident $tch_assign_tensor_method:ident $change:ty => $type:ty $( => $base:ty )?) => {
        impl<S: Shape, D: Device> std::ops::$trait<Tensor<S, D, $type>> for $type {
            type Output = Tensor<S, D, $change>;
            fn $method(self, rhs: Tensor<S, D, $type>) -> Self::Output {
                Tensor {
                    repr: self.$tch_method(&rhs.repr),
                    ..Default::default()
                }
            }
        }
        impl<S: Shape, D: Device> std::ops::$trait<&Tensor<S, D, $type>> for $type {
            type Output = Tensor<S, D, $change>;

            fn $method(self, rhs: &Tensor<S, D, $type>) -> Self::Output {
                Tensor {
                    repr: self.$tch_method(&rhs.repr),
                    ..Default::default()
                }
            }
        }
        impl<S: Shape, D: Device> std::ops::$trait<$type> for Tensor<S, D, $type> {
            type Output = Tensor<S, D, $change>;

            fn $method(self, rhs: $type) -> Self::Output {
                Tensor {
                    repr: self.repr.$tch_tensor_method(rhs $( as $base )? ),
                    ..Default::default()
                }
            }
        }
        impl<S: Shape, D: Device> std::ops::$trait<$type> for &Tensor<S, D, $type> {
            type Output = Tensor<S, D, $change>;

            fn $method(self, rhs: $type) -> Self::Output {
                Tensor {
                    repr: self.repr.$tch_tensor_method(rhs $( as $base )?),
                    ..Default::default()
                }
            }
        }
    };
}

operator!(Add AddAssign add add_assign add g_add_scalar g_add_scalar_ i32 => i64);
operator!(Add AddAssign add add_assign add g_add_scalar g_add_scalar_ i64);
operator!(Add AddAssign add add_assign add g_add_scalar g_add_scalar_ f32 => f64);
operator!(Add AddAssign add add_assign add g_add_scalar g_add_scalar_ f64);

operator!(Sub SubAssign sub sub_assign sub g_sub_scalar g_sub_scalar_ i32 => i64);
operator!(Sub SubAssign sub sub_assign sub g_sub_scalar g_sub_scalar_ i64);
operator!(Sub SubAssign sub sub_assign sub g_sub_scalar g_sub_scalar_ f32 => f64);
operator!(Sub SubAssign sub sub_assign sub g_sub_scalar g_sub_scalar_ f64);

operator!(Mul MulAssign mul mul_assign mul g_mul_scalar g_mul_scalar_ i32 => i64);
operator!(Mul MulAssign mul mul_assign mul g_mul_scalar g_mul_scalar_ i64);
operator!(Mul MulAssign mul mul_assign mul g_mul_scalar g_mul_scalar_ f32 => f64);
operator!(Mul MulAssign mul mul_assign mul g_mul_scalar g_mul_scalar_ f64);

operator!(@change Div DivAssign div div_assign div g_div_scalar g_div_scalar_ f32 =>  i32 => i64);
operator!(@change Div DivAssign div div_assign div g_div_scalar g_div_scalar_ f64 => i64);
operator!(Div DivAssign div div_assign div g_div_scalar g_div_scalar_ f32 => f64);
operator!(Div DivAssign div div_assign div g_div_scalar g_div_scalar_ f64);

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! def_test {
        ($name:ident => $op:tt => $type:ty => $expected:expr) => {
            #[test]
            fn $name() {
                let t1: Tensor<crate::shapes::shape::Rank2<2, 3>, crate::device::Cpu, $type> = Tensor::ones();
                let tmp: Tensor<crate::shapes::shape::Rank2<2, 3>, crate::device::Cpu, $type> = Tensor::ones();
                let t2 = &t1 $op 1 as $type;
                let t3 = 1 as $type $op &t1;
                let t4 = t1 $op 1 as $type;
                let t5 = 1 as $type $op tmp;
                assert_eq!(t2.to_tch().sum(None), $expected);
                assert_eq!(t3.to_tch().sum(None), $expected);
                assert_eq!(t4.to_tch().sum(None), $expected);
                assert_eq!(t5.to_tch().sum(None), $expected);
            }
        };
        (@assign $name:ident => $op:tt => $type:ty => $expected:expr) => {
            #[test]
            fn $name() {
                let mut t1: Tensor<crate::shapes::shape::Rank2<2, 3>, crate::device::Cpu, $type> = Tensor::ones();
                t1 $op 1 as $type;
                assert_eq!(t1.to_tch().sum(None), $expected);
            }
        };
    }

    def_test!(test_add_i32 => + => i32 => tch::Tensor::from(12));
    def_test!(test_add_i64 => + => i64 => tch::Tensor::from(12));
    def_test!(test_add_f32  => + => f32 => tch::Tensor::from(12.0));
    def_test!(test_add_f64  => + => f64 => tch::Tensor::from(12.0));

    def_test!(@assign test_add_assign_i32 => += => i32 => tch::Tensor::from(12));
    def_test!(@assign test_add_assign_i64 => += => i64 => tch::Tensor::from(12));
    def_test!(@assign test_add_assign_f32  => += => f32 => tch::Tensor::from(12.0));
    def_test!(@assign test_add_assign_f64  => += => f64 => tch::Tensor::from(12.0));

    def_test!(test_sub_i32 => - => i32 => tch::Tensor::from(0));
    def_test!(test_sub_i64 => - => i64 => tch::Tensor::from(0));
    def_test!(test_sub_f32  => - => f32 => tch::Tensor::from(0.0));
    def_test!(test_sub_f64  => - => f64 => tch::Tensor::from(0.0));

    def_test!(@assign test_sub_assign_i32 => -= => i32 => tch::Tensor::from(0));
    def_test!(@assign test_sub_assign_i64 => -= => i64 => tch::Tensor::from(0));
    def_test!(@assign test_sub_assign_f32  => -= => f32 => tch::Tensor::from(0.0));
    def_test!(@assign test_sub_assign_f64  => -= => f64 => tch::Tensor::from(0.0));

    def_test!(test_mul_i32 => * => i32 => tch::Tensor::from(6));
    def_test!(test_mul_i64 => * => i64 => tch::Tensor::from(6));
    def_test!(test_mul_f32  => * => f32 => tch::Tensor::from(6.0));
    def_test!(test_mul_f64  => * => f64 => tch::Tensor::from(6.0));

    def_test!(@assign test_mul_assign_i32 => *= => i32 => tch::Tensor::from(6));
    def_test!(@assign test_mul_assign_i64 => *= => i64 => tch::Tensor::from(6));
    def_test!(@assign test_mul_assign_f32  => *= => f32 => tch::Tensor::from(6.0));
    def_test!(@assign test_mul_assign_f64  => *= => f64 => tch::Tensor::from(6.0));

    def_test!(test_div_i32 => / => i32 => tch::Tensor::from(6.0));
    def_test!(test_div_i64 => / => i64 => tch::Tensor::from(6.0));
    def_test!(test_div_f32  => / => f32 => tch::Tensor::from(6.0));
    def_test!(test_div_f64  => / => f64 => tch::Tensor::from(6.0));

    def_test!(@assign test_div_assign_f32  => /= => f32 => tch::Tensor::from(6.0));
    def_test!(@assign test_div_assign_f64  => /= => f64 => tch::Tensor::from(6.0));
}
