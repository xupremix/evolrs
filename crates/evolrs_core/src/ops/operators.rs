// macro_rules! operator {
//     ($trait:ident $method:ident $operator:op $tch_method:ident $($dim:ident)*) => {
//         impl<
//
//
//         > $trait for crate::tensor::Tensor<S, D, K> {
//             fn $method(&self, other: &Self) -> Self {
//                 Self {
//                     repr,
//                     shape:
//                     kind: self.kind,
//                 }
//             }
//         }
//     };
//     (@replace $x:tt $xs:expr) => {$xs};
//     (@count $($x:tt)*) => {<[()]>::len(&[$(operator!(@replace $x ())),*])};
// }
//
// macro_rules! shape {
//     ($Name:ident$(,)? ) => {
//         #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
//         pub struct $Name;
//         shape!(@impl $Name);
//     };
//     ($Name:ident $(, $Dim:ident)+ $(,)?) => {
//         #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
//         pub struct $Name<
//             $(const $Dim: usize,)+
//         >;
//         shape!(@impl $Name $(, $Dim)+);
//     };
//     (@impl $Name:ident $($(, $Dim:ident)+)?) => {
//         impl $(<$(const $Dim: usize,)+>)? Shape for
//             $Name $(<$($Dim,)+>)? {
//             const DIMS: usize = shape!(@count $($($Dim)+)?);
//             const NELEMS: usize = 0 $(+ 1 $( * $Dim)+)?;
//         }
//     };
// }
//
