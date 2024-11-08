use syn::parse::Parse;

pub(crate) struct Args {
    pub(crate) vis: syn::Visibility,
    pub(crate) dims: syn::LitInt,
}

impl Parse for Args {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        Ok(Self {
            vis: input
                .parse()
                .map_err(|e| syn::Error::new(e.span(), "expected visibility identifier"))?,
            dims: input
                .parse()
                .map_err(|e| syn::Error::new(e.span(), "expected dim expression"))?,
        })
    }
}
