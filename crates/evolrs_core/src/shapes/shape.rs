use std::{fmt::Debug, hash::Hash};

pub trait Shape: 'static + Debug + Clone + Copy + Send + Sync + PartialEq + Eq + Hash {
    type ArrayType: 'static + Debug + Clone + Copy + Send + Sync + PartialEq + Eq + Hash;
    const DIMS: usize;
    const NELEMS: usize;
    fn dims() -> &'static [i64];
}

macro_rules! shape {
    ($Name:ident$(,)? ) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub struct $Name;
        shape!(@impl $Name);
    };
    ($Name:ident $(, $Dim:ident)+ $(,)?) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub struct $Name<
            $(const $Dim: usize,)+
        >;
        shape!(@impl $Name $(, $Dim)+);
    };
    (@impl $Name:ident $($(, $Dim:ident)+)?) => {
        impl $(<$(const $Dim: usize,)+>)? Shape for
            $Name $(<$($Dim,)+>)? {
            type ArrayType = shape!(@array $($($Dim)+)?);
            const DIMS: usize = shape!(@count $($($Dim)+)?);
            const NELEMS: usize = 0 $(+ 1 $( * $Dim)+)?;
            fn dims() -> &'static [i64] {
                &[$( $($Dim as i64),* )?]
            }
        }
    };
    (@array) => {
        [usize; 0]
    };
    (@array $x:tt) => {
        [usize; $x]
    };
    (@array $x:tt $($xs:tt)+) => {
        [shape!(@array $($xs)+); $x]
    };
    (@replace $x:tt $xs:expr) => {$xs};
    (@count $($x:tt)*) => {<[()]>::len(&[$(shape!(@replace $x ())),*])};
}

shape!(Scalar);
shape!(Rank1, D0);
shape!(Rank2, D0, D1);
shape!(Rank3, D0, D1, D2);
shape!(Rank4, D0, D1, D2, D3);
shape!(Rank5, D0, D1, D2, D3, D4);
shape!(Rank6, D0, D1, D2, D3, D4, D5);
shape!(Rank7, D0, D1, D2, D3, D4, D5, D6);
shape!(Rank8, D0, D1, D2, D3, D4, D5, D6, D7);

#[cfg(test)]
mod tests {}
