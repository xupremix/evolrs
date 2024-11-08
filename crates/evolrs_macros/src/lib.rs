use proc_macro::TokenStream;

pub(crate) mod crate_shape;
pub(crate) mod module;
pub(crate) mod pub_shape;

#[proc_macro]
pub fn crate_shape(input: TokenStream) -> TokenStream {
    crate_shape::shape(input)
}

#[proc_macro]
pub fn shape(input: TokenStream) -> TokenStream {
    input
}

#[proc_macro_derive(Module)]
pub fn module(input: TokenStream) -> TokenStream {
    input
}
