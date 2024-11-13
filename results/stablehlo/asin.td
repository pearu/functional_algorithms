// This file is generated using functional_algorithms tool (0.11.0), see
//   https://github.com/pearu/functional_algorithms
// for more information.




def : Pat<(asin_0 NonComplexElementType:$z),
  (StableHLO_AddOp
    (StableHLO_Atan2Op:$ta
      $z,
      (StableHLO_AddOp
        (StableHLO_ConstantLike<"1.0">:$one $z),
        (StableHLO_SqrtOp
          (StableHLO_MulOp
            (StableHLO_SubtractOp $one, $z),
            (StableHLO_AddOp $one, $z))))),
    $ta)>;

def : Pat<(asin_1 ComplexElementType:$z),
  (StableHLO_ComplexOp
    (StableHLO_Atan2Op:$real
      (StableHLO_RealOp:$signed_x $z),
      (StableHLO_RealOp
        (CHLO_AsinAcosKernelOp:$asin_acos_kernel_z $z))),
    (StableHLO_SelectOp
      (StableHLO_CompareOp
       (StableHLO_ImagOp:$signed_y $z),
       (StableHLO_ConstantLike<"0.0"> $signed_x),
        StableHLO_ComparisonDirectionValue<"LT">,
        (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
      (StableHLO_NegOp
        (StableHLO_ImagOp:$imag_asin_acos_kernel_z $asin_acos_kernel_z)),
      $imag_asin_acos_kernel_z))>;