use proc_macro2::TokenStream;
use quote::quote;
use syn::Ident;

pub(crate) fn matmul(dims: i64, name: &Ident, dim_idents: &[Ident]) -> TokenStream {
    if dims < 2 {
        return quote! {};
    }

    let all_but_last_two = (0..dims - 2)
        .map(|i| &dim_idents[i as usize])
        .collect::<Vec<_>>();
    let penultimate_of_second = &dim_idents[dims as usize - 2];
    let last_of_first = Ident::new(&format!("N_D{}", dims - 1), proc_macro2::Span::call_site());
    let last_of_second = Ident::new(&format!("D{}", dims - 1), proc_macro2::Span::call_site());

    let const_dims = dim_idents.iter().map(|i| quote! { const #i: usize, });
    let const_last_of_first = quote! { const #last_of_first: usize, };

    quote! {
        impl<
            #(#const_dims)*
            #const_last_of_first
        > crate::tensor::wrap::matmul::Matmul<
            #name < #(#dim_idents,)* >
        > for #name <
            #(#all_but_last_two,)*
            #last_of_second,
            #last_of_first
        > {
            type MatmulShape = #name <
                #(#all_but_last_two,)*
                #penultimate_of_second,
                #last_of_first
            >;
        }
    }
}
