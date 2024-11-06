use std::fmt::Debug;
use std::hash::Hash;

pub trait Device:
    'static + Debug + Clone + Copy + Send + Sync + PartialEq + Eq + Hash + Into<tch::Device>
{
}

macro_rules! device {
    ($n:ident $t:ident $($v:ident)?) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub struct $n $(<const $v: usize>)?;

        impl $(<const $v: usize>)? Device for $n $(<$v>)? {}
        impl $(<const $v: usize>)? From<$n $(<$v>)?> for tch::Device {
            fn from(_: $n $(<$v>)? ) -> tch::Device {
                Self::$t$(($v))?
            }
        }
    };
}

device!(Cpu Cpu);
device!(Cuda Cuda N);
device!(Mps Mps);
device!(Vulkan Vulkan);

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! def_test {
        ($n:ident $t:ident $($v:expr)?) => {
            #[test]
            fn $n() {
                assert_eq!(
                    <$t$(<$v>)? as std::convert::Into<tch::Device>>::into($t),
                    tch::Device::$t$(($v))?
                );
            }
        };
    }

    def_test!(test_cpu Cpu);
    def_test!(test_cuda1 Cuda 0);
    def_test!(test_cuda2 Cuda 42);
    def_test!(test_mps Mps);
    def_test!(test_vulkan Vulkan);
}
