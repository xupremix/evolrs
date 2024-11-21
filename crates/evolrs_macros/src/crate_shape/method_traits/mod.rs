use proc_macro2::TokenStream;
use quote::quote;
use syn::Ident;

#[cfg(feature = "broadcast-semantics")]
mod broadcast;
#[cfg(feature = "broadcast-semantics")]
mod broadcast_inplace;

mod argmax;
mod argmin;
mod matmul;

pub(crate) fn gen_methods(dims: i64, name: &Ident, dim_idents: &[Ident]) -> TokenStream {
    let matmul = matmul::matmul(dims, name, dim_idents);
    let argmax = argmax::argmax(dims, name, dim_idents);
    let argmin = argmin::argmin(dims, name, dim_idents);

    #[cfg(feature = "broadcast-semantics")]
    let broadcast = broadcast::broadcast(dims, name);

    #[cfg(feature = "broadcast-semantics")]
    let broadcast_inplace = broadcast_inplace::broadcast_inplace(dims, name);

    #[cfg(feature = "broadcast-semantics")]
    quote! {
        #matmul
        #argmax
        #argmin
        #broadcast
        #broadcast_inplace
    }
    #[cfg(not(feature = "broadcast-semantics"))]
    quote! {
        #matmul
        #argmax
        #argmin
    }
}
