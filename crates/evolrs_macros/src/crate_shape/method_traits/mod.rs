use proc_macro2::TokenStream;
use quote::quote;
use syn::Ident;

#[cfg(feature = "broadcast-semantics")]
mod broadcast;
mod matmul;

pub(crate) fn gen_methods(dims: i64, name: &Ident, dim_idents: &[Ident]) -> TokenStream {
    let matmul = matmul::matmul(dims, name, dim_idents);

    #[cfg(feature = "broadcast-semantics")]
    let broadcast = broadcast::broadcast(dims, name, dim_idents);

    quote! {
        #matmul
        #broadcast
    }
}
