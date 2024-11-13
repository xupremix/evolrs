use crate::device::Device;
use crate::kind::Kind;
use crate::shapes::shape::Shape;
use crate::tensor::Tensor;

macro_rules! wrap {
    ($( [ $f1:ident - $f2:ident ] ),* $(,)?) => {
        impl<S: Shape, D: Device, K: Kind> Tensor<S, D, K> {
            $(
                pub fn $f1(&self) -> Tensor<S, D, K> {
                    Tensor {
                        repr: self.repr.$f1(),
                        ..Default::default()
                    }
                }
                pub fn $f2(&mut self) -> Tensor<S, D, K> {
                    Tensor {
                        repr: self.repr.$f2(),
                        ..Default::default()
                    }
                }
            )*
        }
    };
}

wrap! {
    [abs - abs_],
    [acos - acos_],
    [arccos - arccos_],
    [asin - asin_],
    [arcsin - arcsin_],
    [cos - cos_],
    [cosh - cosh_],
    [acosh - acosh_],
    [arccosh - arccosh_],
    [detach - detach_],
    [erf - erf_],
    [erfc - erfc_],
    [erfinv - erfinv_],
    [exp - exp_],
    [expm1 - expm1_],
    [fix - fix_],
    [floor - floor_],
    [frac - frac_],
    [i0 - i0_],
    [lgamma - lgamma_],
    [log - log_],
    [log10 - log10_],
    [log1p - log1p_],
    [log2 - log2_],
    [neg - neg_],
    [reciprocal - reciprocal_],
    [round - round_],
    [rsqrt - rsqrt_],
    [sgn - sgn_],
    [sigmoid - sigmoid_],
    [sign - sign_],
    [sin - sin_],
    [sinc - sinc_],
    [sinh - sinh_],
    [asinh - asinh_],
    [arcsinh - arcsinh_],
    [trunc - trunc_],
    [sqrt - sqrt_],
    [square - square_],
    [tan - tan_],
    [atan - atan_],
    [tanh - tanh_],
    [atanh - atanh_],
    [arctan - arctan_],
    [arctanh - arctanh_],

}

// TODO: remove the operations from bool tensors
// TODO: check for operations when S = Scalar

// different:
// - add
// - addbmm
// - addcdiv
// - addcmul
// - addmm
// - sspaddmm
// - addmv
// - adjoint
// - allclose
// - amax
// - amin
// - angle
// - apply
// - argmax
// - argmin
// - argsort
// - argwhere
// - as_strided
// - all
// - any
// - backward
// - baddmm
// - bernoulli
// - bincount
// - bitwise_not
// - bitwise_anderfinv
// - bitwise_or
// - bitwise_xor
// - bitwise_left_shift
// - bitwise_right_shift
// - bmm
// - .type (equivalent to .to_dtype...)
// - broadcast_to (equal to .expand)
// - cauchy
// - ceil
// - cholesky
// - cholesky_inverse
// - cholesky_solve
// - chunk (correlated with tensor_split)
// - tensor_split (correlated with chunk)
// - clamp
// - clip
// - clone? (check impl in mod.rs)
// - copy (check impl)
// - conj (_physical / resolve_conj / resolve_neg)
// - copysign
// - corrcoef
// - count_nonzero
// - cov
// - cross
// - logcumsumexp
// - cummax / cummin / cumprod / cumsum
// - deg2rad
// - dequantize
// - det
// - dense_dim
// - detach
// - diag
// - diag_embed
// - diagflat
// - diagonal
// - diagonal_scatter
// - fill_diagonal_
// - fmax / fmin
// - diff
// - digamma
// - dim (S::DIMS)
// - dim_order
// - dist
// - div
// - dot
// - dsplit
// - element_size
// - eq
// - equal
// - expand
// - expand_as (by conventions it should be renamed to expand_like)
// - exponential_
// - flip
// - fliplr
// - flipud
// - float_power
// - floor_divide
// - fmod
// - frexp
// - gather
// - gcd
// - ge
// - geometric_
// - geqrf
// - ger (same as outer)
// - gt
// - heaviside
// - histc
// - histogram
// - hsplit
// - hypot
// - igamma (same as gammainc)
// - igammac (same as gammaincc)
// - index_add
// - index_copy
// - index_fill
// - index_put
// - index_reduce
// - index_select
// - indices
// - inner
// - int_repr (geiven a quantized tensor returns a CPU Tensor with uint8)
// - inv
// - isclose
// - isfinite
// - isinf
// - isposinf
// - isneginf
// - isnan
// - is_continuous
// - is_conj
// - is_inference
// - is_leaf
// - is_pinned
// - is_set_to
// - is_shared
// - is_signed
// - is_sparse
// - istft
// - is_real
// - item (implemented as int64_value / double_value)
// - kthvalue
// - lcm
// - ldexp
// - le
// - lerp
// - logdet
// - log_normal_
// - logaddexp
// - logaddexp2
// - logsumexp
// - logical_and
// - logical_not
// - logical_or
// - logical_xor
// - logit
// - lt
// - lu_factor
// - lu_solve
// - map
// - masked_scatter
// - masked_fill
// - masked_select
// - matmul (already impl check for hidden behavior like with broadcasting)
// - matrix_power
// - matrix_exp
// - max
// - maximum
// - mean
// - module_load
// - nanmean
// - median
// - nanmedian
// - min
// - mm
// - smm
// - mode
// - movedim
// - moveaxis
// - msort
// - mul
// - multiply
// - multinomial
// - mv
// - mvlgamma (same as multigammaln)
// - narrow
// - nansum
// - narrow_copy
// - nan_to_num
// - ne
// - not_equal
// - numel (S::NUMEL)
// - nextafter
// - nonzero
// - norm (Deprecated, use vector_norm / matrix_norm / linalg.norm)
// - orgqr
// - ormqr
// - outer (previously linked)
// - permute
// - pin_memory
// - pinverse (same as pinv)
// - polygamma
// - positive
// - pow
// - qr
// - put_
// - prod
// - qscheme
// - quantile
// - nanquantile
// - q_scale
// - q_zero_point
// - q_per_challel_scales
// - q_per_channel_zero_points
// - q_per_channel_axis
// - rad2deg
// - random (should be alrady impl)
// - ravel (same as flatten)
// - record_stream
// - register_hook
// - register_post_accumulate_grad_hood
// - remainder
// - renorm
// - repeat
// - repeat_interleave
// - requires_grad
// - reshape
// - reshape_as (reshape_like)
// - resize
// - retain_grad
// - retains_grad
// - roll
// - rot90
// - scatter
// - scatter_add
// - scatter_reduce
// - select
// - select_scatter
// - set
// - share_memory_
// - signbit
// - shape
// - slogdet
// - slice_scatter
// - sort
// - split
// - sparse_mask
// - sparse_dim
// - squeeze
// - std
// - stft
// (storage stuff)
// - stride
// - sub
// - sum
// - sum_to_size
// - svd
// - swapaxes
// - swapdims
// - t
// - tensor_split (previously linked with chunk)
// - tile
// - to (split impl for .to_dev and .to_dtyp)
// - to_mkldnn
// - take
// - take_along_dim
// - tolist
// - to_dense
// - to_sparse
// - to_sparse_csr
// - to_sparse_csc
// - to_sparse_bsr
// - to_sparse_bsc
// - trace
// - transpose
// - triangular_solve
// - tril
// - triu
// - true_divide
// - trunc
// - unbind
// - unflatten
// - unfold
// - uniform
// - unique
// - unique_consecutive
// - unsqueeze
// - values
// - var
// - vdot
// - view
// - view_as (view_like)
// - vsplit
// - where
// - xlogy
// - xpu
