use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::Ident;

pub(crate) fn unsqueeze(dims: i64, name: &Ident, dim_idents: &[Ident]) -> TokenStream {
    if dims == 0 {
        return quote! {};
    }
    if dims == 1 {
        return quote! {
            impl crate::tensor::wrap::unsqueeze::Unsqueeze<0>
                for Scalar {
                type UnsqueezeShape = Rank1<1>;
            }
        };
    }

    let mut out = vec![];

    let dims = dims as usize;
    for i in 0..dims {
        let mut bits = vec![false; dims];
        bits[i] = true;

        let new_name = Ident::new(&format!("Rank{}", dims - 1), Span::call_site());

        let mut idents = dim_idents.iter().collect::<Vec<_>>();
        let _ = idents.pop();
        let const_dims = idents
            .iter()
            .map(|d| quote! { const #d: usize })
            .collect::<Vec<_>>();

        let mut count = 0;
        let new_idents = bits
            .iter()
            .map(|&b| {
                if b {
                    quote! {
                        1usize
                    }
                } else {
                    let i = &idents[count];
                    count += 1;
                    quote! {
                        #i
                    }
                }
            })
            .collect::<Vec<_>>();

        out.push(quote! {
            impl<
               #(#const_dims),*
            > crate::tensor::wrap::unsqueeze::Unsqueeze<#i>
            for #new_name<#(#idents),*> {
                type UnsqueezeShape = #name<#(#new_idents),*>;
            }
        });
    }

    quote! {
        #(#out)*
    }
}
