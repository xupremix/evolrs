use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::Ident;

pub(crate) fn broadcast(dims: i64, name: &Ident, dim_idents: &[Ident]) -> TokenStream {
    if dims < 2 {
        return quote! {};
    }

    // given a dimension dims > 0
    // broadcast requirements:
    // When iterating over the dimension sizes, starting at the trailing dimension, the dimension sizes must either be equal, one of them is 1, or one of them does not exist.
    //
    // given curr_dim and dims
    // broadcast should be implemented for:
    // - curr_dim until curr_dim <= dims
    //
    // Example:
    // given dims = 3
    // broadcast should be implemented for:
    // dims = 1
    // dims = 2
    // dims = 3
    //
    // Example:
    // given dims = 4
    // broadcast should be implemented for:
    // dims = 1
    // dims = 2
    // dims = 3
    // dims = 4

    let mut toks = vec![];
    let mut curr_dim = 1;
    while curr_dim <= dims {
        let shape_curr = Ident::new(&format!("Rank{}", curr_dim), Span::call_site());
        let shape_gen = name;

        let const_dims_curr = (0..curr_dim)
            .map(|i| {
                let i = Ident::new(&format!("D{}", i), Span::call_site());
                quote! { const #i: usize, }
            })
            .collect::<Vec<_>>();
        let const_dims_gen = (0..dims)
            .map(|i| {
                let i = Ident::new(&format!("G_D{}", i), Span::call_site());
                quote! { const #i: usize, }
            })
            .collect::<Vec<_>>();

        let const_dims = quote! {
            #(#const_dims_curr)*
            #(#const_dims_gen)*
        };

        let idents_curr = (0..curr_dim)
            .map(|i| Ident::new(&format!("D{}", i), Span::call_site()))
            .collect::<Vec<_>>();

        let idents_gen = (0..dims)
            .map(|i| Ident::new(&format!("G_D{}", i), Span::call_site()))
            .collect::<Vec<_>>();

        let assert_msg = format!(
            "Broadcasting is not possible for {} and {}",
            shape_curr, shape_gen
        );

        let assert_check_gen = gen_assert_check(curr_dim, dims, true);

        toks.push(quote! {
            impl<
                #const_dims
                > crate::ops::method_traits::broadcast::Broadcast<
                    #shape_gen < #(#idents_gen),* >
                > for #shape_curr < #(#idents_curr),* > {
                const CHECK: () = assert!(#assert_check_gen, #assert_msg);
                type BroadcastShape = #shape_gen < #(#idents_gen),* >;
            }
        });
        if curr_dim != dims {
            let assert_check_curr = gen_assert_check(curr_dim, dims, false);
            toks.push(quote! {
                impl<
                    #const_dims
                    > crate::ops::method_traits::broadcast::Broadcast<
                    #shape_curr < #(#idents_curr),* >
                    > for #shape_gen < #(#idents_gen),* > {
                    const CHECK: () = assert!(#assert_check_curr, #assert_msg);
                    type BroadcastShape = #shape_curr < #(#idents_curr),* >;
                }
            });
        }
        curr_dim += 1;
    }
    quote! {
        #(#toks)*
    }
}

fn gen_assert_check(mut curr_dim: i64, mut dims: i64, gen: bool) -> TokenStream {
    // Reminder:
    // broadcast requirements:
    // When iterating over the dimension sizes, starting at the trailing dimension the dimension sizes must either be:
    // - equal,
    // - one of them is 1
    // - or one of them does not exist.
    let mut toks = vec![];
    while dims > 0 {
        if curr_dim == 0 && dims > 0 {
            break;
        }
        let dim_curr = Ident::new(&format!("D{}", curr_dim - 1), Span::call_site());
        let dim_gen = Ident::new(&format!("G_D{}", dims - 1), Span::call_site());
        if gen {
            toks.push(quote! {
                (#dim_curr == #dim_gen || #dim_gen == 1 ) &&
            });
        } else {
            toks.push(quote! {
                (#dim_curr == #dim_gen || #dim_curr == 1 ) &&
            });
        }
        curr_dim -= 1;
        dims -= 1;
    }
    quote! {
        #(#toks)* true
    }
}
