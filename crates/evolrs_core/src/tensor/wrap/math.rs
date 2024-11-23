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
