// This file is generated using functional_algorithms tool (0.4.0), see
//   https://github.com/pearu/functional_algorithms
// for more information.




def : Pat<(acos_0 NonComplexElementType:$z),
  (StableHLO_MulOp
    (StableHLO_ConstantLike<"2"> $z),
    (StableHLO_Atan2Op
      (StableHLO_SqrtOp
        (StableHLO_MulOp
          (StableHLO_SubtractOp
            (StableHLO_ConstantLike<"1">:$one $z),
            $z),
          (StableHLO_AddOp:$add_one_z $one, $z))),
      $add_one_z))>;

def : Pat<(acos_1 ComplexElementType:$z),
  (StableHLO_ComplexOp
    (StableHLO_Atan2Op
      (StableHLO_RealOp
        (CHLO_AsinAcosKernelOp:$asin_acos_kernel_z $z)),
      (StableHLO_RealOp:$signed_x $z)),
    (StableHLO_SelectOp
      (StableHLO_CompareOp
       (StableHLO_ImagOp:$signed_y $z),
       (StableHLO_ConstantLike<"0"> $signed_y),
        StableHLO_ComparisonDirectionValue<"LT">,
        (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
      (StableHLO_ImagOp:$imag_asin_acos_kernel_z $asin_acos_kernel_z),
      (StableHLO_NegOp $imag_asin_acos_kernel_z)))>;