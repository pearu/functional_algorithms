// This file is generated using functional_algorithms tool (0.15.1.dev3+ge93b47e.d20250113), see
//   https://github.com/pearu/functional_algorithms
// for more information.




def : Pat<(acosh_0 NonComplexElementType:$z),
  (StableHLO_SelectOp
    (StableHLO_CompareOp
     $z,
     (StableHLO_DivOp
       (StableHLO_ConstantLikeMaxFiniteValue $z),
       (StableHLO_ConstantLike<"2.0">:$constant_f2 $z)),
      StableHLO_ComparisonDirectionValue<"GE">,
      (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
    (StableHLO_AddOp
      (StableHLO_LogOp $constant_f2),
      (StableHLO_LogOp $z)),
    (StableHLO_Log1pOp
      (StableHLO_MulOp
        (StableHLO_SqrtOp:$sqrt_26
          (StableHLO_SubtractOp
            $z,
            (StableHLO_ConstantLike<"1.0">:$constant_f1 $z))),
        (StableHLO_AddOp
          (StableHLO_SqrtOp
            (StableHLO_AddOp $z, $constant_f1)),
          $sqrt_26))))>;

def : Pat<(acosh_1 ComplexElementType:$z),
  (StableHLO_ComplexOp
    (StableHLO_ImagOp
      (CHLO_AsinAcosKernelOp:$w $z)),
    (StableHLO_SelectOp
      (StableHLO_CompareOp
       (StableHLO_ImagOp:$signed_y $z),
       (StableHLO_ConstantLike<"0.0"> $signed_y),
        StableHLO_ComparisonDirectionValue<"LT">,
        (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
      (StableHLO_NegOp
        (StableHLO_Atan2Op:$imag
          (StableHLO_RealOp $w),
          (StableHLO_RealOp:$signed_x $z))),
      $imag))>;