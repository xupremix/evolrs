use crate::{
    device::Device,
    kind::Kind,
    shapes::shape::Shape,
    tensor::{RequiresGrad, Tensor},
};

pub trait View<Src: Shape>: Shape {
    const VIEW_CHECK: ();
}

impl<S: Shape, D: Shape> View<S> for D {
    const VIEW_CHECK: () = assert!(
        S::NELEMS == D::NELEMS,
        "The two shapes do not have the same number of elements"
    );
}

// TODO: check if view_as inherits gradient tracking

impl<S: Shape, D: Device, K: Kind, G: RequiresGrad> Tensor<S, D, K, G> {
    pub fn view<S2: View<S>>(&self) -> Tensor<S2, D, K, G> {
        #![allow(path_statements)]
        S2::VIEW_CHECK;
        Tensor {
            repr: self.repr.view(S2::dims()),
            ..Default::default()
        }
    }

    pub fn view_copy<S2: View<S>>(&self) -> Tensor<S2, D, K, G> {
        #![allow(path_statements)]
        S2::VIEW_CHECK;
        Tensor {
            repr: self.repr.view_copy(S2::dims()),
            ..Default::default()
        }
    }

    pub fn view_as<S2: View<S>, D2: Device, K2: Kind>(
        &self,
        other: &Tensor<S2, D2, K2, G>,
    ) -> Tensor<S2, D, K, G> {
        #![allow(path_statements)]
        S2::VIEW_CHECK;
        Tensor {
            repr: self.repr.view_as(&other.repr),
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {}
