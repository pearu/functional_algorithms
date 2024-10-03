// This file is generated using functional_algorithms tool (0.10.2.dev9+g7001467.d20241002), see
//   https://github.com/pearu/functional_algorithms
// for more information.




def : Pat<(acos_0 NonComplexElementType:$z),
  (StableHLO_Atan2Op
    (StableHLO_SqrtOp
      (StableHLO_MulOp
        (StableHLO_SubtractOp
          (StableHLO_ConstantLike<"1.0">:$constant_f1 $z),
          $z),
        (StableHLO_AddOp $constant_f1, $z))),
    $z)>;

def : Pat<(acos_1 ComplexElementType:$z),
  (StableHLO_ComplexOp
    (StableHLO_Atan2Op
      (StableHLO_RealOp
        (CHLO_AsinAcosKernelOp:$asin_acos_kernel_z $z)),
      (StableHLO_RealOp:$signed_x $z)),
    (StableHLO_SelectOp
      (StableHLO_CompareOp
       (StableHLO_ImagOp:$signed_y $z),
       (StableHLO_ConstantLike<"0.0"> $signed_y),
        StableHLO_ComparisonDirectionValue<"LT">,
        (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
      (StableHLO_ImagOp:$imag_asin_acos_kernel_z $asin_acos_kernel_z),
      (StableHLO_NegOp $imag_asin_acos_kernel_z)))>;