use crate::{
    device::Device,
    kind::{qi32, qi8, qu8, Kind},
    shapes::shape::Shape,
    tensor::Tensor,
};

pub trait Item<S: Shape, T: Kind> {
    const ITEM_CHECK: () = assert!(
        S::NELEMS == 1,
        "Item can only be called on a tensor with one element"
    );
    fn item(&self) -> T;
}

macro_rules! def_item {
    ($type:ty => $fn:ident $( $t:ident ),* $(,)?) => {
        $(
            impl<S: Shape, D: Device> Item<S, $type> for Tensor<S, D, $t> {
                fn item(&self) -> $type {
                    #![allow(path_statements)]
                    Self::ITEM_CHECK;
                    let zv = vec![0; self.dims() as usize];
                    self.repr.$fn(&zv)
                }
            }
        )*
    };
}

def_item!(f64 => double_value f32, f64);
def_item!(i64 => int64_value u8, i8, i16, i32, i64, qi8, qu8, qi32);
