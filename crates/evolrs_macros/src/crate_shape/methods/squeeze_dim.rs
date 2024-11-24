use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::Ident;

pub(crate) fn squeeze_dim(dims: i64, name: &Ident, dim_idents: &[Ident]) -> TokenStream {
    if dims == 0 {
        return quote! {};
    }
    if dims == 1 {
        return quote! {
            impl crate::tensor::wrap::squeeze::SqueezeDim<0>
                for Rank1<1> {
                type SqueezeShape = Scalar;
            }
        };
    }

    let mut out = vec![];

    let dims = dims as usize;
    for i in 0..dims {
        let mut bits = vec![false; dims];
        bits[i] = true;

        let new_name = Ident::new(&format!("Rank{}", dims - 1), Span::call_site());
        let mut count = 0;

        let const_dims = bits
            .iter()
            .filter_map(|&b| {
                if b {
                    None
                } else {
                    let i = &dim_idents[count];
                    count += 1;
                    Some(quote! {
                        const #i: usize
                    })
                }
            })
            .collect::<Vec<_>>();
        count = 0;
        let idents = bits
            .iter()
            .map(|&b| {
                if b {
                    quote! {
                        1usize
                    }
                } else {
                    let i = &dim_idents[count];
                    count += 1;
                    quote! {
                        #i
                    }
                }
            })
            .collect::<Vec<_>>();
        count = 0;
        let new_idents = bits
            .iter()
            .filter_map(|&b| {
                if b {
                    None
                } else {
                    let i = &dim_idents[count];
                    count += 1;
                    Some(quote! {
                        #i
                    })
                }
            })
            .collect::<Vec<_>>();

        out.push(quote! {
            impl<
               #(#const_dims),*
            > crate::tensor::wrap::squeeze::SqueezeDim<#i>
            for #name<#(#idents),*> {
                type SqueezeShape = #new_name<#(#new_idents),*>;
            }
        });
    }

    quote! {
        #(#out)*
    }
}
