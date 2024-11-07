use crate::{device::Device, kind::Kind, shapes::shape::Shape, tensor::Tensor};

macro_rules! operator {
    ($trait:ident $assign_trait:ident $method:ident $assign_method:ident $tch_method:ident $tch_assign_method:ident) => {
        impl<S: Shape, D: Device, K: Kind> std::ops::$trait for Tensor<S, D, K> {
            type Output = Self;
            fn $method(self, rhs: Self) -> Self::Output {
                Self {
                    repr: self.repr.$tch_method(&rhs.repr),
                    ..Default::default()
                }
            }
        }
        impl<S: Shape, D: Device, K: Kind> std::ops::$trait<&Self> for Tensor<S, D, K> {
            type Output = Self;
            fn $method(self, rhs: &Self) -> Self::Output {
                Self {
                    repr: self.repr.$tch_method(&rhs.repr),
                    ..Default::default()
                }
            }
        }
        impl<S: Shape, D: Device, K: Kind> std::ops::$trait for &Tensor<S, D, K> {
            type Output = Tensor<S, D, K>;

            fn $method(self, rhs: Self) -> Self::Output {
                Tensor {
                    repr: self.repr.$tch_method(&rhs.repr),
                    ..Default::default()
                }
            }
        }
        impl<S: Shape, D: Device, K: Kind> std::ops::$trait<Tensor<S, D, K>> for &Tensor<S, D, K> {
            type Output = Tensor<S, D, K>;

            fn $method(self, rhs: Tensor<S, D, K>) -> Self::Output {
                Tensor {
                    repr: self.repr.$tch_method(&rhs.repr),
                    ..Default::default()
                }
            }
        }
        impl<S: Shape, D: Device, K: Kind> std::ops::$assign_trait for Tensor<S, D, K> {
            fn $assign_method(&mut self, rhs: Self) {
                let _ = self.repr.$tch_assign_method(&rhs.repr);
            }
        }
        impl<S: Shape, D: Device, K: Kind> std::ops::$assign_trait<&Self> for Tensor<S, D, K> {
            fn $assign_method(&mut self, rhs: &Self) {
                let _ = self.repr.$tch_assign_method(&rhs.repr);
            }
        }
    };
    (@div $trait:ident $assign_trait:ident $method:ident $assign_method:ident $tch_method:ident $tch_assign_method:ident $type:ty) => {
        impl<S: Shape, D: Device> std::ops::$trait for Tensor<S, D, $type> {
            type Output = Tensor<S, D, $type>;
            fn $method(self, rhs: Self) -> Self::Output {
                Tensor {
                    repr: self.repr.$tch_method(&rhs.repr),
                    ..Default::default()
                }
            }
        }
        impl<S: Shape, D: Device> std::ops::$trait<&Self> for Tensor<S, D, $type> {
            type Output = Tensor<S, D, $type>;
            fn $method(self, rhs: &Self) -> Self::Output {
                Tensor {
                    repr: self.repr.$tch_method(&rhs.repr),
                    ..Default::default()
                }
            }
        }
        impl<S: Shape, D: Device> std::ops::$trait for &Tensor<S, D, $type> {
            type Output = Tensor<S, D, $type>;

            fn $method(self, rhs: Self) -> Self::Output {
                Tensor {
                    repr: self.repr.$tch_method(&rhs.repr),
                    ..Default::default()
                }
            }
        }
        impl<S: Shape, D: Device> std::ops::$trait<Tensor<S, D, $type>> for &Tensor<S, D, $type> {
            type Output = Tensor<S, D, $type>;

            fn $method(self, rhs: Tensor<S, D, $type>) -> Self::Output {
                Tensor {
                    repr: self.repr.$tch_method(&rhs.repr),
                    ..Default::default()
                }
            }
        }
        impl<S: Shape, D: Device> std::ops::$assign_trait for Tensor<S, D, $type> {
            fn $assign_method(&mut self, rhs: Self) {
                let _ = self.repr.$tch_assign_method(&rhs.repr);
            }
        }
        impl<S: Shape, D: Device> std::ops::$assign_trait<&Self> for Tensor<S, D, $type> {
            fn $assign_method(&mut self, rhs: &Self) {
                let _ = self.repr.$tch_assign_method(&rhs.repr);
            }
        }
    };
    (@change $trait:ident $assign_trait:ident $method:ident $assign_method:ident $tch_method:ident $tch_assign_method:ident $type:ty => $change:ty) => {
        impl<S: Shape, D: Device> std::ops::$trait for Tensor<S, D, $type> {
            type Output = Tensor<S, D, $change>;
            fn $method(self, rhs: Self) -> Self::Output {
                Tensor {
                    repr: self.repr.$tch_method(&rhs.repr),
                    ..Default::default()
                }
            }
        }
        impl<S: Shape, D: Device> std::ops::$trait<&Self> for Tensor<S, D, $type> {
            type Output = Tensor<S, D, $change>;
            fn $method(self, rhs: &Self) -> Self::Output {
                Tensor {
                    repr: self.repr.$tch_method(&rhs.repr),
                    ..Default::default()
                }
            }
        }
        impl<S: Shape, D: Device> std::ops::$trait for &Tensor<S, D, $type> {
            type Output = Tensor<S, D, $change>;

            fn $method(self, rhs: Self) -> Self::Output {
                Tensor {
                    repr: self.repr.$tch_method(&rhs.repr),
                    ..Default::default()
                }
            }
        }
        impl<S: Shape, D: Device> std::ops::$trait<Tensor<S, D, $type>> for &Tensor<S, D, $type> {
            type Output = Tensor<S, D, $change>;

            fn $method(self, rhs: Tensor<S, D, $type>) -> Self::Output {
                Tensor {
                    repr: self.repr.$tch_method(&rhs.repr),
                    ..Default::default()
                }
            }
        }
    };
}

operator!(Add AddAssign add add_assign g_add g_add_);
operator!(Sub SubAssign sub sub_assign g_sub g_sub_);
operator!(Mul MulAssign mul mul_assign g_mul g_mul_);
operator!(@div Div DivAssign div div_assign g_div g_div_ f32);
operator!(@div Div DivAssign div div_assign g_div g_div_ f64);
operator!(@change Div DivAssign div div_assign g_div g_div_ i32 => f32);
operator!(@change Div DivAssign div div_assign g_div g_div_ i64 => f64);

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! def_test {
        ($name:ident => $op:tt => $expected:expr => $type:ty => $change:ty) => {
            #[test]
            fn $name() {
                let t1: Tensor<crate::shapes::shape::Rank2<2, 3>, crate::device::Cpu, $type> = Tensor::ones();
                let t2: Tensor<crate::shapes::shape::Rank2<2, 3>, crate::device::Cpu, $type> = Tensor::ones();
                let t3: Tensor<crate::shapes::shape::Rank2<2, 3>, crate::device::Cpu, $change> = t1 $op t2;
                assert_eq!(t3.to_tch().sum(None), $expected);

                let t1: Tensor<crate::shapes::shape::Rank2<2, 3>, crate::device::Cpu, $type> = Tensor::ones();
                let t2: Tensor<crate::shapes::shape::Rank2<2, 3>, crate::device::Cpu, $type> = Tensor::ones();
                let t3: Tensor<crate::shapes::shape::Rank2<2, 3>, crate::device::Cpu, $change> = &t1 $op t2;
                assert_eq!(t3.to_tch().sum(None), $expected);

                let t1: Tensor<crate::shapes::shape::Rank2<2, 3>, crate::device::Cpu, $type> = Tensor::ones();
                let t2: Tensor<crate::shapes::shape::Rank2<2, 3>, crate::device::Cpu, $type> = Tensor::ones();
                let t3: Tensor<crate::shapes::shape::Rank2<2, 3>, crate::device::Cpu, $change> = t1 $op &t2;
                assert_eq!(t3.to_tch().sum(None), $expected);

                let t1: Tensor<crate::shapes::shape::Rank2<2, 3>, crate::device::Cpu, $type> = Tensor::ones();
                let t2: Tensor<crate::shapes::shape::Rank2<2, 3>, crate::device::Cpu, $type> = Tensor::ones();
                let t3: Tensor<crate::shapes::shape::Rank2<2, 3>, crate::device::Cpu, $change> = &t1 $op &t2;
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
    def_test!(test_div_i64 => / => tch::Tensor::from(6.0) => i64 => f64);
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
