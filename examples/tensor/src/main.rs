use evolrs::{device::Cpu, shapes::shape::Rank2, tch, tensor::Tensor};

fn main() {
    macro_rules! new_test {
        ($lhs_ty:ident $rhs_ty:ident => $n1:tt => $n2:tt) => {
            let t1 = evolrs::tch::Tensor::ones([2, 3], (tch::Kind::$lhs_ty, tch::Device::Cpu));
            let t2 = evolrs::tch::Tensor::ones([2, 3], (tch::Kind::$rhs_ty, tch::Device::Cpu));
            let t3 = t1 / t2;
            println!("{} / {} = {:?}", $n1, $n2, t3.kind());
        };
    }
    macro_rules! permuatations {
        ($l:ident => $ltt:tt => $r:ident => $rtt:tt $(=> $l2:ident => $l2tt:tt)+) => {
            new_test!($l $r => $ltt => $rtt);
            new_test!($l $l => $ltt => $ltt);
            new_test!($r $l => $rtt => $ltt);
            new_test!($r $r => $rtt => $rtt);
            permuatations!($l => $ltt $(=> $l2 => $l2tt)+);
        };
        ($l:ident => $ltt:tt => $r:ident => $rtt:tt) => {
            new_test!($l $r => $ltt => $rtt);
            new_test!($l $l => $ltt => $ltt);
            new_test!($r $l => $rtt => $ltt);
            new_test!($r $r => $rtt => $rtt);
        };
    }
    // create all the possible lhs and rhs combinations

    permuatations!(Uint8 => "Uint8" => Int8 => "Int8" => Int16 => "Int16" => Int => "Int" => Int64 => "Int64" => Half => "Half" => Float => "Float" => Double => "Double" => ComplexFloat => "ComplexFloat" => ComplexDouble => "ComplexDouble" );
    // permuatations!(Int8 => "Int8" => Int16 => "Int16" => Int => "Int" => Int64 => "Int64" => Half => "Half" => Float => "Float" => Double => "Double" => ComplexHalf => "ComplexHalf" => ComplexFloat => "ComplexFloat" => ComplexDouble => "ComplexDouble" => Bool => "Bool" );
    // permuatations!(Int16 => "Int16" => Int => "Int" => Int64 => "Int64" => Half => "Half" => Float => "Float" => Double => "Double" => ComplexHalf => "ComplexHalf" => ComplexFloat => "ComplexFloat" => ComplexDouble => "ComplexDouble" => Bool => "Bool" );
    // permuatations!(Int => "Int" => Int64 => "Int64" => Half => "Half" => Float => "Float" => Double => "Double" => ComplexHalf => "ComplexHalf" => ComplexFloat => "ComplexFloat" => ComplexDouble => "ComplexDouble" => Bool => "Bool" );
    // permuatations!(Int64 => "Int64" => Half => "Half" => Float => "Float" => Double => "Double" => ComplexHalf => "ComplexHalf" => ComplexFloat => "ComplexFloat" => ComplexDouble => "ComplexDouble" => Bool => "Bool" );
    // permuatations!(Half => "Half" => Float => "Float" => Double => "Double" => ComplexHalf => "ComplexHalf" => ComplexFloat => "ComplexFloat" => ComplexDouble => "ComplexDouble" => Bool => "Bool" );
    // permuatations!(Float => "Float" => Double => "Double" => ComplexHalf => "ComplexHalf" => ComplexFloat => "ComplexFloat" => ComplexDouble => "ComplexDouble" => Bool => "Bool" );
    // permuatations!(Double => "Double" => ComplexHalf => "ComplexHalf" => ComplexFloat => "ComplexFloat" => ComplexDouble => "ComplexDouble" => Bool => "Bool" );
    // permuatations!(ComplexHalf => "ComplexHalf" => ComplexFloat => "ComplexFloat" => ComplexDouble => "ComplexDouble" => Bool => "Bool" );
    // permuatations!(ComplexFloat => "ComplexFloat" => ComplexDouble => "ComplexDouble" => Bool => "Bool" );
    // permuatations!(ComplexDouble => "ComplexDouble" => Bool => "Bool" );

    /*
    Uint8,
    Int8,
    Int16,
    Int,
    Int64,
    Half,
    Float,
    Double,
    ComplexHalf,
    ComplexFloat,
    ComplexDouble,
    Bool,
    QInt8,
    QUInt8,
    QInt32,
    BFloat16,
    */
}
