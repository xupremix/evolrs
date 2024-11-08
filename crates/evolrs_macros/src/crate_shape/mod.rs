use parse_args::Args;
use proc_macro2::TokenStream;
use quote::quote;
use syn::Ident;

mod method_traits;
mod parse_args;

pub(crate) fn shape(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let args = syn::parse_macro_input!(input as Args);
    let vis = args.vis;
    let dims: usize = args.dims.base10_parse().expect("expected integer literal");

    let name = if dims == 0 {
        Ident::new("Scalar", proc_macro2::Span::call_site())
    } else {
        Ident::new(&format!("Rank{}", dims), proc_macro2::Span::call_site())
    };

    let dim_idents = (0..dims)
        .map(|i| Ident::new(&format!("D{}", i), proc_macro2::Span::call_site()))
        .collect::<Vec<_>>();
    let nelems = if dims == 0 {
        quote! { 0 }
    } else {
        quote! { 1 #(* #dim_idents)* }
    };
    let (const_generics, generics) = generics(dims, &dim_idents);
    let shape = if dims == 0 {
        quote! { [usize; 0] }
    } else {
        gen_shape(dims - 1, 0)
    };

    let method_traits = method_traits::gen_methods(dims, &name, &dim_idents);

    quote! {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        #vis struct #name #const_generics;

        impl #const_generics Shape for #name #generics {
            type Shape = #shape;
            const DIMS: usize = #dims;
            const NELEMS: usize = #nelems;
            fn dims() -> &'static [i64] {
                &[#( #dim_idents as i64 ),*]
            }
        }

        #method_traits
    }
    .into()
}

fn gen_shape(dims: usize, curr: usize) -> TokenStream {
    let dim = Ident::new(&format!("D{}", curr), proc_macro2::Span::call_site());
    if dims == 0 {
        return quote! { [usize; #dim] };
    }
    let shape = gen_shape(dims - 1, curr + 1);
    quote! {
        [#shape; #dim]
    }
}

fn generics(dims: usize, dim_idents: &[Ident]) -> (TokenStream, TokenStream) {
    if dims == 0 {
        return (quote! {}, quote! {});
    }
    (
        quote! {
            <
                #(const #dim_idents: usize),*
            >
        },
        quote! {
            <
                #(#dim_idents),*
            >
        },
    )
}

// pub trait Shape: 'static + Debug + Clone + Copy + Send + Sync + PartialEq + Eq + Hash {
//     type Shape: 'static + Debug + Clone + Copy + Send + Sync + PartialEq + Eq + Hash;
//     const DIMS: usize;
//     const NELEMS: usize;
//     fn dims() -> &'static [i64];
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
//             type Shape = shape!(@array $($($Dim)+)?);
//             const DIMS: usize = shape!(@count $($($Dim)+)?);
//             const NELEMS: usize = 0 $(+ 1 $( * $Dim)+)?;
//             fn dims() -> &'static [i64] {
//                 &[$( $($Dim as i64),* )?]
//             }
//         }
//     };
//     (@array) => {
//         [usize; 0]
//     };
//     (@array $x:tt) => {
//         [usize; $x]
//     };
//     (@array $x:tt $($xs:tt)+) => {
//         [shape!(@array $($xs)+); $x]
//     };
//     (@replace $x:tt $xs:expr) => {$xs};
//     (@count $($x:tt)*) => {<[()]>::len(&[$(shape!(@replace $x ())),*])};
// }
