pub mod argmax;
pub mod argmin;
pub mod cmp;
pub mod flatten;
pub mod func;
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
// Also check what methods inherit gradient tracking and which don't
//
// TLDR:
// ( Extra generic params )
// - Requires Gradient
// - Initialization
// - Memory Format
// - Pin
//
// ( Implementation )
// implementation of from for tensor for arrays
// without using `unsafe from_blob`
//
// Losses: _loss (or divergence) functions
// L1
// MSE
// CrossEntropy
// CTC
// NLL
// PoissonNLL
// GaussianNLL (Not present)
// KL
// BCE
// BCEWithLogits
// MarginRanking
// HingeEmbedding
// MultiLabelMargin
// Huber
// SmoothL1
// SoftMargin
// MultiLabelSoftMargin
// CosineEmbedding
// MultiMargin
// TripletMargin
// TripleMarginWithDistance (Not present)
