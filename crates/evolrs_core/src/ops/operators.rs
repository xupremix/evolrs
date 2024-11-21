use crate::{
    device::Device,
    kind::{
        type_coercion::{Coerce, DivCoerce, Same},
        Kind,
    },
    shapes::shape::Shape,
    tensor::Tensor,
};

macro_rules! op {
    ($( $trait:ident $trait_:ident $fn:ident $tch_fn:ident $fn_:ident $tch_fn_:ident ),* $(,)?) => {
        $(
            impl<S: Shape, D: Device, K: Kind, K2: Coerce<K>> std::ops::$trait<Tensor<S, D, K2>>
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
            impl<S: Shape, D: Device, K: Kind, K2: Coerce<K>> std::ops::$trait<&Tensor<S, D, K2>>
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
            impl<S: Shape, D: Device, K: Kind, K2: Coerce<K>> std::ops::$trait<Tensor<S, D, K2>>
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
            impl<S: Shape, D: Device, K: Kind, K2: Coerce<K>> std::ops::$trait<&Tensor<S, D, K2>>
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
            impl<S: Shape, D: Device, K: Kind, K2: Coerce<K>> std::ops::$trait_<Tensor<S, D, K2>> for Tensor<S, D, K>
            where
                K: Same<K2::To>
            {
                fn $fn_(&mut self, rhs: Tensor<S, D, K2>) {
                    let _ = self.repr.$tch_fn_(&rhs.repr);
                }
            }
            impl<S: Shape, D: Device, K: Kind, K2: Coerce<K>> std::ops::$trait_<&Tensor<S, D, K2>> for Tensor<S, D, K>
            where
                K: Same<K2::To>
            {
                fn $fn_(&mut self, rhs: &Tensor<S, D, K2>) {
                    let _ = self.repr.$tch_fn_(&rhs.repr);
                }
            }
        )*
    };
    (@div) => {
        impl<S: Shape, D: Device, K: Kind, K2: DivCoerce<K>> std::ops::Div<Tensor<S, D, K2>>
            for Tensor<S, D, K>
        {
            type Output = Tensor<S, D, K2::To>;
            fn div(self, rhs: Tensor<S, D, K2>) -> Self::Output {
                Tensor {
                    repr: self.repr.g_div(&rhs.repr),
                    ..Default::default()
                }
            }
        }
        impl<S: Shape, D: Device, K: Kind, K2: DivCoerce<K>> std::ops::Div<&Tensor<S, D, K2>>
            for Tensor<S, D, K>
        {
            type Output = Tensor<S, D, K2::To>;
            fn div(self, rhs: &Tensor<S, D, K2>) -> Self::Output {
                Tensor {
                    repr: self.repr.g_div(&rhs.repr),
                    ..Default::default()
                }
            }
        }
        impl<S: Shape, D: Device, K: Kind, K2: DivCoerce<K>> std::ops::Div<Tensor<S, D, K2>>
            for &Tensor<S, D, K>
        {
            type Output = Tensor<S, D, K2::To>;
            fn div(self, rhs: Tensor<S, D, K2>) -> Self::Output {
                Tensor {
                    repr: self.repr.g_div(&rhs.repr),
                    ..Default::default()
                }
            }
        }
        impl<S: Shape, D: Device, K: Kind, K2: DivCoerce<K>> std::ops::Div<&Tensor<S, D, K2>>
            for &Tensor<S, D, K>
        {
            type Output = Tensor<S, D, K2::To>;
            fn div(self, rhs: &Tensor<S, D, K2>) -> Self::Output {
                Tensor {
                    repr: self.repr.g_div(&rhs.repr),
                    ..Default::default()
                }
            }
        }
        impl<
            S: Shape,
            D: Device,
            K: Kind,
            K2: DivCoerce<K>
        > std::ops::DivAssign<
            Tensor<S, D, K2>
        > for Tensor<S, D, K>
        where
            K: Same<K2::To>
        {
            fn div_assign(&mut self, rhs: Tensor<S, D, K2>) {
                let _ = self.repr.g_div_(&rhs.repr);
            }
        }
    };
}

op! {
   Add AddAssign add g_add add_assign g_add_,
   Sub SubAssign sub g_sub sub_assign g_sub_,
   Mul MulAssign mul g_mul mul_assign g_mul_,
}
op! {
    @div
}

//
// #[cfg(test)]
// mod tests {
//     use super::*;
//
//     macro_rules! def_test {
//         ($name:ident => $op:tt => $expected:expr => $type:ty => $change:ty) => {
//             #[test]
//             fn $name() {
//                 let t1: Tensor<crate::shapes::shape::Rank2<2, 3>, crate::device::Cpu, $type> = Tensor::ones();
//                 let t2: Tensor<crate::shapes::shape::Rank2<2, 3>, crate::device::Cpu, $type> = Tensor::ones();
//                 let t3: Tensor<crate::shapes::shape::Rank2<2, 3>, crate::device::Cpu, $change> = t1 $op t2;
//                 assert_eq!(t3.to_tch().sum(None), $expected);
//
//                 let t1: Tensor<crate::shapes::shape::Rank2<2, 3>, crate::device::Cpu, $type> = Tensor::ones();
//                 let t2: Tensor<crate::shapes::shape::Rank2<2, 3>, crate::device::Cpu, $type> = Tensor::ones();
//                 let t3: Tensor<crate::shapes::shape::Rank2<2, 3>, crate::device::Cpu, $change> = &t1 $op t2;
//                 assert_eq!(t3.to_tch().sum(None), $expected);
//
//                 let t1: Tensor<crate::shapes::shape::Rank2<2, 3>, crate::device::Cpu, $type> = Tensor::ones();
//                 let t2: Tensor<crate::shapes::shape::Rank2<2, 3>, crate::device::Cpu, $type> = Tensor::ones();
//                 let t3: Tensor<crate::shapes::shape::Rank2<2, 3>, crate::device::Cpu, $change> = t1 $op &t2;
//                 assert_eq!(t3.to_tch().sum(None), $expected);
//
//                 let t1: Tensor<crate::shapes::shape::Rank2<2, 3>, crate::device::Cpu, $type> = Tensor::ones();
//                 let t2: Tensor<crate::shapes::shape::Rank2<2, 3>, crate::device::Cpu, $type> = Tensor::ones();
//                 let t3: Tensor<crate::shapes::shape::Rank2<2, 3>, crate::device::Cpu, $change> = &t1 $op &t2;
//                 assert_eq!(t3.to_tch().sum(None), $expected);
//             }
//         };
//         (@assign $name:ident => $op:tt => $expected:expr) => {
//             #[test]
//             fn $name() {
//                 let mut t1: Tensor<crate::shapes::shape::Rank2<2, 3>> = Tensor::ones();
//                 let t2: Tensor<crate::shapes::shape::Rank2<2, 3>> = Tensor::ones();
//                 t1 $op t2;
//                 assert_eq!(t1.to_tch().sum(None), $expected);
//             }
//         };
//     }
//
//     def_test!(test_add_i32 => + => tch::Tensor::from(12) => i32 => i32);
//     def_test!(test_add_i64 => + => tch::Tensor::from(12) => i64 => i64);
//     def_test!(test_add_f32 => + => tch::Tensor::from(12.0) => f32 => f32);
//     def_test!(test_add_f64 => + => tch::Tensor::from(12.0) => f64 => f64);
//
//     def_test!(test_sub_i32 => - => tch::Tensor::from(0) => i32 => i32);
//     def_test!(test_sub_i64 => - => tch::Tensor::from(0) => i64 => i64);
//     def_test!(test_sub_f32 => - => tch::Tensor::from(0.0) => f32 => f32);
//     def_test!(test_sub_f64 => - => tch::Tensor::from(0.0) => f64 => f64);
//
//     def_test!(test_mul_i32 => * => tch::Tensor::from(6) => i32 => i32);
//     def_test!(test_mul_i64 => * => tch::Tensor::from(6) => i64 => i64);
//     def_test!(test_mul_f32 => * => tch::Tensor::from(6.0) => f32 => f32);
//     def_test!(test_mul_f64 => * => tch::Tensor::from(6.0) => f64 => f64);
//
//     def_test!(test_div_i32 => / => tch::Tensor::from(6.0) => i32 => f32);
//     def_test!(test_div_i64 => / => tch::Tensor::from(6.0) => i64 => f64);
//     def_test!(test_div_f32 => / => tch::Tensor::from(6.0) => f32 => f32);
//     def_test!(test_div_f64 => / => tch::Tensor::from(6.0) => f64 => f64);
//
//     def_test!(@assign test_add_assign_i32 => += => tch::Tensor::from(12));
//     def_test!(@assign test_add_assign_i64 => += => tch::Tensor::from(12));
//     def_test!(@assign test_add_assign_f32 => += => tch::Tensor::from(12.0));
//     def_test!(@assign test_add_assign_f64 => += => tch::Tensor::from(12.0));
//
//     def_test!(@assign test_sub_assign_i32 => -= => tch::Tensor::from(0));
//     def_test!(@assign test_sub_assign_i64 => -= => tch::Tensor::from(0));
//     def_test!(@assign test_sub_assign_f32 => -= => tch::Tensor::from(0.0));
//     def_test!(@assign test_sub_assign_f64 => -= => tch::Tensor::from(0.0));
//
//     def_test!(@assign test_mul_assign_i32 => *= => tch::Tensor::from(6));
//     def_test!(@assign test_mul_assign_i64 => *= => tch::Tensor::from(6));
//     def_test!(@assign test_mul_assign_f32 => *= => tch::Tensor::from(6.0));
//     def_test!(@assign test_mul_assign_f64 => *= => tch::Tensor::from(6.0));
//
//     def_test!(@assign test_div_assign_f32 => /= => tch::Tensor::from(6.0));
//     def_test!(@assign test_div_assign_f64 => /= => tch::Tensor::from(6.0));
// }
