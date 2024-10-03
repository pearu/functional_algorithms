// This file is generated using functional_algorithms tool (0.10.2.dev9+g7001467.d20241002), see
//   https://github.com/pearu/functional_algorithms
// for more information.




def : Pat<(asinh_0 NonComplexElementType:$z),
  (StableHLO_MulOp
    (StableHLO_SignOp $z),
    (StableHLO_SelectOp
      (StableHLO_CompareOp
       (StableHLO_AbsOp:$ax $z),
       (StableHLO_SqrtOp
         (StableHLO_ConstantLikeMaxFiniteValue $z)),
        StableHLO_ComparisonDirectionValue<"GE">,
        (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
      (StableHLO_AddOp
        (StableHLO_LogOp
          (StableHLO_ConstantLike<"2.0"> $z)),
        (StableHLO_LogOp $ax)),
      (StableHLO_Log1pOp
        (StableHLO_AddOp
          $ax,
          (StableHLO_DivOp
            (StableHLO_MulOp:$ax2 $ax, $ax),
            (StableHLO_AddOp
              (StableHLO_ConstantLike<"1.0">:$one $z),
              (StableHLO_SqrtOp
                (StableHLO_AddOp $one, $ax2))))))))>;

def : Pat<(asinh_1 ComplexElementType:$z),
  (StableHLO_ComplexOp
    (StableHLO_SelectOp
      (StableHLO_CompareOp
       (StableHLO_RealOp:$signed_x $z),
       (StableHLO_ConstantLike<"0.0"> $signed_x),
        StableHLO_ComparisonDirectionValue<"LT">,
        (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
      (StableHLO_NegOp
        (StableHLO_ImagOp:$imag_w
          (CHLO_AsinAcosKernelOp:$w
            (StableHLO_ComplexOp
              (StableHLO_NegOp
                (StableHLO_ImagOp:$signed_y $z)),
              $signed_x)))),
      $imag_w),
    (StableHLO_Atan2Op
      $signed_y,
      (StableHLO_RealOp $w)))>;