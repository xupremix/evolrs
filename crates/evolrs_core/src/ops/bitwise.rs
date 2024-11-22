use crate::{
    device::Device,
    kind::{
        restriction::composite::{IntOrBool, NotBool},
        type_coercion::{Coerce, Same},
    },
    shapes::shape::Shape,
    tensor::Tensor,
};

macro_rules! op {
    ($( $trait:ident $trait_:ident $fn:ident $tch_fn:ident $fn_:ident $tch_fn_:ident ),* $(,)?) => {
        $(
            impl<S: Shape, D: Device, K: IntOrBool, K2: Coerce<K> + IntOrBool> std::ops::$trait<Tensor<S, D, K2>>
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
            impl<S: Shape, D: Device, K: IntOrBool, K2: Coerce<K> + IntOrBool> std::ops::$trait<&Tensor<S, D, K2>>
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
            impl<S: Shape, D: Device, K: IntOrBool, K2: Coerce<K> + IntOrBool> std::ops::$trait<Tensor<S, D, K2>>
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
            impl<S: Shape, D: Device, K: IntOrBool, K2: Coerce<K> + IntOrBool> std::ops::$trait<&Tensor<S, D, K2>>
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
            impl<S: Shape, D: Device, K: IntOrBool, K2: Coerce<K> + IntOrBool> std::ops::$trait_<Tensor<S, D, K2>> for Tensor<S, D, K>
            where
                K: Same<K2::To>
            {
                fn $fn_(&mut self, rhs: Tensor<S, D, K2>) {
                    let _ = self.repr.$tch_fn_(&rhs.repr);
                }
            }
            impl<S: Shape, D: Device, K: IntOrBool, K2: Coerce<K> + IntOrBool> std::ops::$trait_<&Tensor<S, D, K2>> for Tensor<S, D, K>
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

// TODO: Broadcasting impls
op! {
   BitAnd BitAndAssign bitand bitwise_and_tensor bitand_assign bitwise_and_tensor_,
   BitOr BitOrAssign bitor bitwise_or_tensor bitor_assign bitwise_or_tensor_,
   BitXor BitXorAssign bitxor bitwise_xor_tensor bitxor_assign bitwise_xor_tensor_,
   Shl ShlAssign shl bitwise_left_shift shl_assign bitwise_left_shift_,
   Shr ShrAssign shr bitwise_right_shift shr_assign bitwise_right_shift_,
}

#[cfg(test)]
mod tests {}
