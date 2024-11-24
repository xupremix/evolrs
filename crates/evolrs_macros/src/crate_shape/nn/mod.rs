use proc_macro2::TokenStream;
use quote::quote;
use syn::Ident;

mod forward;

pub(crate) fn methods(dims: i64, name: &Ident, dim_idents: &[Ident]) -> TokenStream {
    let forward = forward::forward(dims, name, dim_idents);
    quote! {
        #forward
    }
}
