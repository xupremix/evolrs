use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::Ident;

pub(crate) fn forward(dims: i64, name: &Ident, dim_idents: &[Ident]) -> TokenStream {
    if dims < 1 {
        return quote! {};
    }

    let const_generics = (0..dims + 1)
        .map(|d| {
            let d = Ident::new(&format!("D{}", d), Span::call_site());
            quote! { const #d: usize }
        })
        .collect::<Vec<_>>();

    let mut all_idents = dim_idents.to_vec();
    all_idents.push(Ident::new(&format!("D{}", dims), Span::call_site()));

    let dims = dims as usize;
    let forward_idents = [&all_idents[dims - 1], &all_idents[dims]];
    let mut new_idents = all_idents.clone();
    new_idents.remove(new_idents.len() - 2);

    quote! {
        impl <
            #(#const_generics),*
        > crate::nn::Forward<#(#forward_idents),*> for #name<#(#dim_idents),*> {
            type ForwardShape = #name<#(#new_idents),*>;
        }
    }
}
