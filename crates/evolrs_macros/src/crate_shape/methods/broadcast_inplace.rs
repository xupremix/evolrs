use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::Ident;

pub(crate) fn broadcast_inplace(dims: i64, name: &Ident) -> TokenStream {
    if dims < 1 {
        return quote! {};
    }

    let mut toks = vec![];
    let mut curr_dim = 1;
    while curr_dim <= dims {
        let shape_curr = Ident::new(&format!("Rank{}", curr_dim), Span::call_site());
        let shape_gen = name;

        // Generation of const dims
        let const_dims = (0..curr_dim)
            .map(|i| {
                let i = Ident::new(&format!("D{}", i), Span::call_site());
                quote! { const #i: usize, }
            })
            .collect::<Vec<_>>();
        let const_dims_g = (0..dims)
            .map(|i| {
                let i = Ident::new(&format!("G_D{}", i), Span::call_site());
                quote! { const #i: usize, }
            })
            .collect::<Vec<_>>();

        let const_dims = quote! {
            #(#const_dims)*
            #(#const_dims_g)*
        };

        // Generation of idents
        let idents = (0..curr_dim)
            .map(|i| Ident::new(&format!("D{}", i), Span::call_site()))
            .collect::<Vec<_>>();
        let idents_g = (0..dims)
            .map(|i| Ident::new(&format!("G_D{}", i), Span::call_site()))
            .collect::<Vec<_>>();

        let assert_msg = format!(
            "\nThe dimension provided for broadcasting {shape_curr} into {shape_gen} are not compatible.\nTo Broadcast when iterating over the dimension sizes, starting at the trailing dimension, the dimension sizes must either be:\n - equal\n - one of them is 1\n - one of them does not exist\n"
        );

        let assert_check = gen_assert_check(curr_dim, dims);
        toks.push(quote! {
            impl<
                #const_dims
            > crate::shapes::broadcast::BroadcastInplace<
                #shape_gen < #(#idents_g),* >
            > for #shape_curr < #(#idents),* > {
                const BROADCAST_INPLACE_CHECK: () = assert!(#assert_check, #assert_msg);
            }
        });
        curr_dim += 1;
    }
    quote! {
        #(#toks)*
    }
}

fn gen_assert_check(mut curr_dim: i64, mut dims: i64) -> TokenStream {
    // Reminder:
    // broadcast requirements:
    // When iterating over the dimension sizes, starting at the trailing dimension the dimension sizes must either be:
    // - equal,
    // - one of them is 1
    // - or one of them does not exist.
    let mut toks = vec![];
    while curr_dim > 0 {
        let dim_g = Ident::new(&format!("G_D{}", dims - 1), Span::call_site());
        let dim = Ident::new(&format!("D{}", curr_dim - 1), Span::call_site());
        toks.push(quote! {
            (#dim == #dim_g || #dim == 1) &&
        });
        curr_dim -= 1;
        dims -= 1;
    }
    quote! {
        #(#toks)* true
    }
}
