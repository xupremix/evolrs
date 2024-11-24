use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::Ident;

pub(crate) fn squeeze(dims: i64, name: &Ident, dim_idents: &[Ident]) -> TokenStream {
    if dims == 0 {
        return quote! {};
    }

    let mut out = vec![];

    let ones = (0..dims).map(|_| 1).collect::<Vec<usize>>();
    out.push(quote! {
        impl crate::tensor::wrap::squeeze::Squeeze<#name<#(#ones),*>> for Scalar {
            const CHECK: () = assert!(true);
        }
    });

    let const_dims = dim_idents
        .iter()
        .map(|d| quote! { const #d: usize })
        .collect::<Vec<_>>();

    for new_dim in 1..dims {
        let new_name = Ident::new(&format!("Rank{}", new_dim), Span::call_site());
        let new_const_dims = (0..new_dim)
            .map(|d| {
                let i = dims + d;
                let i = Ident::new(&format!("D{}", i), Span::call_site());
                quote! { const #i: usize }
            })
            .collect::<Vec<_>>();
        let new_dims = (0..new_dim)
            .map(|d| {
                let i = dims + d;
                let i = Ident::new(&format!("D{}", i), Span::call_site());
                quote! { #i }
            })
            .collect::<Vec<_>>();
        let const_dims = quote! {
            #(#const_dims,)*
            #(#new_const_dims),*
        };

        let assert_check = gen_assert_check(dims, new_dim);

        out.push(quote! {
            impl<
                #const_dims
            > crate::tensor::wrap::squeeze::Squeeze <
                #name<#(#dim_idents),*>
            > for #new_name<#(#new_dims),*> {
                const CHECK: () = assert!(#assert_check, "Wrong dimensions provided");
            }
        });
    }

    quote! {
        #(#out)*
    }
}

fn gen_assert_check(unsqueezed_dims: i64, squeezed_dims: i64) -> TokenStream {
    let idents = (0..unsqueezed_dims + squeezed_dims)
        .map(|i| {
            let i = Ident::new(&format!("D{}", i), Span::call_site());
            quote! { #i }
        })
        .collect::<Vec<_>>();

    let mut out = vec![];

    let perms = 2u32.pow(unsqueezed_dims as u32) - 1;
    let mut bits = vec![false; unsqueezed_dims as usize];
    let unsqueezed_dims = unsqueezed_dims as usize;
    let perms = perms as usize;
    let min_ones_for_valid_perm = unsqueezed_dims - squeezed_dims as usize;
    for perm in 1..=perms - 1 {
        for i in 0..unsqueezed_dims {
            bits[unsqueezed_dims - 1 - i] = (perm >> i) & 1 == 1
        }

        // Divided in 2 parts:
        // - the and conditions where all Idents where the bit is true they must be 1
        // - chained to another and condition of all other Idents where the bit is set to false
        // being equal to the squeezed dims
        //
        // Ex:
        //    A B C D - E F
        //    1 0 1 0 - - -
        // Becomes:
        // A == 1 &&
        // B == E &&
        // C == 1 &&
        // D == F
        // Note that valid permutations must have at least k bits where k is the distance between
        // the 2 dimension sizes
        if bits.iter().filter(|&&b| b).count() < min_ones_for_valid_perm {
            continue;
        }

        let mut count = 0;
        let cond = bits
            .iter()
            .enumerate()
            .map(|(i, &b)| {
                let unsqueezed_ident = &idents[i];
                if b {
                    quote! { #unsqueezed_ident == 1 }
                } else {
                    let squeezed_ident = &idents[unsqueezed_dims + count];
                    count += 1;
                    quote! { #unsqueezed_ident == #squeezed_ident }
                }
            })
            .collect::<Vec<_>>();

        out.push(quote! { ( #(#cond &&)* true ) });
    }
    quote! {
        #(#out || )* false
    }
}
