pub(crate) trait Sealed {}

pub trait Same<T> {}
impl<T> Same<T> for T {}
