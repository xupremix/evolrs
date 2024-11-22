use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::Ident;

pub(crate) fn argmax(dims: i64, name: &Ident, dim_idents: &[Ident]) -> TokenStream {
    if dims == 1 {
        return quote! {
            impl<
                const D0: usize,
            > crate::tensor::wrap::argmax::ArgmaxShape<0, false> for Rank1<D0> {
                type ArgmaxShape = Scalar;
            }
            impl<
                const D0: usize,
            > crate::tensor::wrap::argmax::ArgmaxShape<0, true> for Rank1<D0> {
                type ArgmaxShape = Rank1<1>;
            }
        };
    }
    let const_dims = (0..dims)
        .map(|i| {
            let i = Ident::new(&format!("D{}", i), Span::call_site());
            quote! { const #i: usize, }
        })
        .collect::<Vec<_>>();
    let mut out = vec![];
    for b in [false, true] {
        for i in 0..dims {
            if b {
                let new_dim_idents = (0..dims)
                    .map(|j| {
                        if i == j {
                            quote! { 1, }
                        } else {
                            let i = &dim_idents[j as usize];
                            quote! { #i, }
                        }
                    })
                    .collect::<Vec<_>>();
                out.push(quote! {
                    impl<
                        #(#const_dims)*
                    > crate::tensor::wrap::argmax::ArgmaxShape<#i, true> for #name<#(#dim_idents),*> {
                        type ArgmaxShape = #name<#(#new_dim_idents)*>;
                    }
                });
            } else {
                let new_name =
                    Ident::new(&format!("Rank{}", dims - 1), proc_macro2::Span::call_site());
                let new_dim_idents = (0..dims)
                    .filter(|&j| j != i)
                    .map(|j| {
                        let i = &dim_idents[j as usize];
                        quote! { #i, }
                    })
                    .collect::<Vec<_>>();
                out.push(quote! {
                    impl<
                        #(#const_dims)*
                    > crate::tensor::wrap::argmax::ArgmaxShape<#i, false> for #name<#(#dim_idents),*> {
                        type ArgmaxShape = #new_name<#(#new_dim_idents)*>;
                    }
                });
            }
        }
    }
    quote! {
        #(#out)*
    }
}
