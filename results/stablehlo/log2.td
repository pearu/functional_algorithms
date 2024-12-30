// This file is generated using functional_algorithms tool (0.13.3.dev1+g8d134ad.d20241230), see
//   https://github.com/pearu/functional_algorithms
// for more information.




def : Pat<(log2_0 ComplexElementType:$z),
  (StableHLO_ComplexOp
    (StableHLO_DivOp
      (StableHLO_RealOp:$x
        (StableHLO_LogOp:$lnz $z)),
      (StableHLO_LogOp:$ln2
        (StableHLO_ConstantLike<"2.0"> $x))),
    (StableHLO_DivOp
      (StableHLO_ImagOp $lnz),
      $ln2))>;