pub mod method_traits;

#[cfg(not(feature = "type-coercion"))]
pub mod operators;
pub mod scalar_operators;

#[cfg(feature = "type-coercion")]
pub mod operators_with_coercion;
