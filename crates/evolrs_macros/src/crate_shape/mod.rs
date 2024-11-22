use parse_args::Args;
use proc_macro2::TokenStream;
use quote::quote;
use syn::Ident;

mod methods;
mod parse_args;

pub(crate) fn shape(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let args = syn::parse_macro_input!(input as Args);
    let dims: i64 = args.dims.base10_parse().expect("expected integer literal");

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

    let method_traits = methods::methods(dims, &name, &dim_idents);

    quote! {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub struct #name #const_generics;

        impl #const_generics Shape for #name #generics {
            type Shape = #shape;
            const DIMS: i64 = #dims;
            const NELEMS: usize = #nelems;
            fn dims() -> &'static [i64] {
                &[#( #dim_idents as i64 ),*]
            }
        }

        #method_traits
    }
    .into()
}

fn gen_shape(dims: i64, curr: usize) -> TokenStream {
    let dim = Ident::new(&format!("D{}", curr), proc_macro2::Span::call_site());
    if dims == 0 {
        return quote! { [usize; #dim] };
    }
    let shape = gen_shape(dims - 1, curr + 1);
    quote! {
        [#shape; #dim]
    }
}

fn generics(dims: i64, dim_idents: &[Ident]) -> (TokenStream, TokenStream) {
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
