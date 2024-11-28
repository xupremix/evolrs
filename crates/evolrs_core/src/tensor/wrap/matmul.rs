use crate::{device::Device, kind::Kind, shapes::shape::Shape, tensor::Tensor};

pub trait Matmul<Rhs: Shape>: Shape {
    type MatmulShape: Shape;
}

// TODO: check if matmul inherits gradient tracking

impl<S: Shape, D: Device, K: Kind> Tensor<S, D, K> {
    pub fn matmul<Rhs: Matmul<S>>(
        &self,
        rhs: &Tensor<Rhs, D, K>,
    ) -> Tensor<Rhs::MatmulShape, D, K> {
        Tensor {
            repr: self.repr.matmul(&rhs.repr),
            ..Default::default()
        }
    }
}

///```compile_fail
/// use crate::evolrs_core::{shapes::shape::Rank3, tensor::Tensor};
/// let t1: Tensor<Rank3<1, 2, 3>> = Tensor::ones();
/// let t2: Tensor<Rank3<1, 3, 4>> = Tensor::ones();
/// let _: Tensor<Rank3<1, 2, 3>> = t1.matmul(&t2);
///```
///
///```compile_fail
/// use crate::evolrs_core::{shapes::shape::Rank3, tensor::Tensor};
/// let t1: Tensor<Rank3<1, 2, 3>> = Tensor::ones();
/// let t2: Tensor<Rank3<1, 3, 4>> = Tensor::ones();
/// let _: Tensor<Rank3<2, 2, 4>> = t1.matmul(&t2);
///```
///
///```compile_fail
/// use crate::evolrs_core::{shapes::shape::Rank3, tensor::Tensor};
/// let t1: Tensor<Rank3<1, 2, 3>> = Tensor::ones();
/// let t2: Tensor<Rank3<1, 3, 4>> = Tensor::ones();
/// let _: Tensor<Rank3<1, 3, 4>> = t1.matmul(&t2);
///```
mod comptime_fails_matmul {}

#[cfg(test)]
mod tests {
    use crate::{
        shapes::shape::Rank2,
        tensor::{Tensor, ToTchTensor as _},
    };

    #[test]
    fn test_matmul() {
        let t1: Tensor<Rank2<2, 3>> = Tensor::ones();
        let t2: Tensor<Rank2<3, 4>> = Tensor::ones();
        let t3 = t1.matmul(&t2);
        assert!(t3.to_tch().equal(&tch::Tensor::from_slice2(&[
            [3., 3., 3., 3.],
            [3., 3., 3., 3.]
        ])));
    }
}
