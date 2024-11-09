use proc_macro2::TokenStream;
use quote::quote;
use syn::Ident;

mod matmul;

pub(crate) fn gen_methods(dims: i64, name: &Ident, dim_idents: &[Ident]) -> TokenStream {
    let matmul = matmul::matmul(dims, name, dim_idents);
    quote! {
        #matmul
    }
}
