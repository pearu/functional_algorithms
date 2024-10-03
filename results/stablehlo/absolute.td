// This file is generated using functional_algorithms tool (0.10.2.dev9+g7001467.d20241002), see
//   https://github.com/pearu/functional_algorithms
// for more information.




def : Pat<(absolute_0 ComplexElementType:$z),
  (StableHLO_SelectOp
    (StableHLO_CompareOp
     (StableHLO_MaxOp:$mx
       (StableHLO_AbsOp:$abs_x
         (StableHLO_RealOp:$x $z)),
       (StableHLO_AbsOp:$abs_y
         (StableHLO_ImagOp $z))),
     (StableHLO_MinOp:$mn $abs_x, $abs_y),
      StableHLO_ComparisonDirectionValue<"EQ">,
      (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
    (StableHLO_MulOp
      (StableHLO_ConstantLike<"1.4142135623730951"> $x),
      $mx),
    (StableHLO_SelectOp
      (StableHLO_AndOp
        (StableHLO_CompareOp
         (StableHLO_SqrtOp:$sqa
           (StableHLO_AddOp
             (StableHLO_ConstantLike<"1.0">:$constant_f1 $x),
             (StableHLO_MulOp:$r
               (StableHLO_DivOp:$mn_over_mx $mn, $mx),
               $mn_over_mx))),
         $constant_f1,
          StableHLO_ComparisonDirectionValue<"EQ">,
          (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
        (StableHLO_CompareOp
         $r,
         (StableHLO_ConstantLike<"0.0"> $x),
          StableHLO_ComparisonDirectionValue<"GT">,
          (STABLEHLO_DEFAULT_COMPARISON_TYPE))),
      (StableHLO_AddOp
        $mx,
        (StableHLO_DivOp
          (StableHLO_MulOp $mx, $r),
          (StableHLO_ConstantLike<"2.0"> $x))),
      (StableHLO_MulOp $mx, $sqa)))>;