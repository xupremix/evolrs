pub mod argmax;
pub mod argmin;
pub mod cmp;
pub mod flatten;
pub mod item;
pub mod math;
pub mod matmul;
pub mod ops;
pub mod scalar_ops;
pub mod squeeze;
pub mod sum;
pub mod unsqueeze;
pub mod view;

// TODO:
// Review the implemented methods and check if they require an initialized
// tensor and if that is a repeating pattern than there should be an extra
// generic for keeping track of the initialization of the tensor at compile time
// same thing goes for methods which require gradient tracking, maybe even
// a way to check if the tensor is currently memory pinned and cannot be moved.
// Then look into how to encapsulate the stride into the type of the tensor
// or how to import from the Shape::Shape type without the usage of
// unsafe and taking ownership of the data without dropping it.
// Another type parameter could be its memory format (eg. contiguous)
//
// TLDR:
// ( Extra generic params )
// - Memory Format
// - Initialization
// - Requires Gradient
// - Pin
//
// ( Implementation )
// implementation of from for tensor for arrays
// without using `unsafe from_blob`
