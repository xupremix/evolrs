pub mod method_traits;

#[cfg(not(feature = "broadcast-semantics"))]
pub mod operators;
#[cfg(not(feature = "broadcast-semantics"))]
pub mod scalar_operators;

#[cfg(feature = "broadcast-semantics")]
pub mod operators_broadcast;
#[cfg(feature = "broadcast-semantics")]
pub mod scalar_operators_broadcast;
