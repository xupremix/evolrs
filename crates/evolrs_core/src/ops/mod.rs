pub mod method_traits;

#[cfg(not(feature = "type-coercion"))]
pub mod operators;
#[cfg(not(feature = "type-coercion"))]
pub mod scalar_operators;

#[cfg(feature = "type-coercion")]
pub mod operators_with_coercion;
#[cfg(feature = "type-coercion")]
pub mod scalar_operators_with_coercion;
