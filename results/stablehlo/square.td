// This file is generated using functional_algorithms tool (0.10.2.dev9+g7001467.d20241002), see
//   https://github.com/pearu/functional_algorithms
// for more information.




def : Pat<(square_0 NonComplexElementType:$z),
  (StableHLO_MulOp $z, $z)>;

def : Pat<(square_1 ComplexElementType:$z),
  (StableHLO_ComplexOp
    (StableHLO_SelectOp
      (StableHLO_AndOp
        (StableHLO_IsFiniteOp
          (StableHLO_RealOp:$x $z)),
        (StableHLO_CompareOp
         (StableHLO_AbsOp $x),
         (StableHLO_AbsOp
           (StableHLO_ImagOp:$y $z)),
          StableHLO_ComparisonDirectionValue<"EQ">,
          (STABLEHLO_DEFAULT_COMPARISON_TYPE))),
      (StableHLO_ConstantLike<"0.0"> $x),
      (StableHLO_MulOp
        (StableHLO_SubtractOp $x, $y),
        (StableHLO_AddOp $x, $y))),
    (StableHLO_MulOp
      (StableHLO_ConstantLike<"2.0"> $x),
      (StableHLO_MulOp $x, $y)))>;