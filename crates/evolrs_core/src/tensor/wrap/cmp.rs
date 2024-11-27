use crate::{
    device::Device,
    kind::{scalar::IntoScalar, Kind},
    shapes::{broadcast::Broadcast, shape::Shape},
    tensor::Tensor,
};

macro_rules! cmp {
    ($( $n_t:ident $tch_t:ident $( $n_s:ident $tch_s:ident )?);* $(;)?) => {
        $(
            impl<S: Shape, D: Device, K: Kind> Tensor<S, D, K> {
                pub fn $n_t<Dst: Shape, Rhs: Broadcast<S, Rhs>, K2: Kind>(
                    &self,
                    rhs: Tensor<Rhs, D, K2>,
                ) -> Tensor<Dst, D, bool> {
                    #![allow(path_statements)]
                    Rhs::BROADCAST_CHECK;
                    Tensor {
                        repr: self.repr.$tch_t(&rhs.repr),
                        ..Default::default()
                    }
                }
            }

            $(
                impl<S: Shape, D: Device, K: Kind> Tensor<S, D, K> {
                    pub fn $n_s<Rhs: IntoScalar>(
                        &self,
                        rhs: Rhs,
                    ) -> Tensor<S, D, bool> {
                        Tensor {
                            repr: self.repr.$tch_s(rhs.into()),
                            ..Default::default()
                        }
                    }
                }
            )?
        )*
    };
}

cmp!(
    lt_t lt_tensor lt lt;
    le_t le_tensor le le;
    gt_t lt_tensor gt lt;
    ge_t le_tensor ge le;
    logical_or logical_or;
    logical_xor logical_xor;
    logical_and logical_and;
);

impl<S: Shape, D: Device, K: Kind> Tensor<S, D, K> {
    pub fn logical_not(&self) -> Tensor<S, D, bool> {
        Tensor {
            repr: self.repr.logical_not(),
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {}
