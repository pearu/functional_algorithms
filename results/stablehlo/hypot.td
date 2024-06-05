// This file is generated using functional_algorithms tool (0.1.2.dev7+g332df57.d20240604), see
//   https://github.com/pearu/functional_algorithms
// for more information.




def : Pat<(hypot_0 NonComplexElementType:$x, NonComplexElementType:$y),
  (StableHLO_SelectOp
    (StableHLO_CompareOp
     (StableHLO_MaxOp:$mx
       (StableHLO_AbsOp:$abs_x $x),
       (StableHLO_AbsOp:$abs_y $y)),
     (StableHLO_MinOp:$mn $abs_x, $abs_y),
      StableHLO_ComparisonDirectionValue<"EQ">,
      (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
    (StableHLO_MulOp
      (StableHLO_SqrtOp
        (StableHLO_ConstantLike<"2">:$constant_2 $x)),
      $mx),
    (StableHLO_SelectOp
      (StableHLO_AndOp
        (StableHLO_CompareOp
         (StableHLO_SqrtOp:$sqa
           (StableHLO_AddOp
             (StableHLO_ConstantLike<"1">:$constant_1 $x),
             (StableHLO_MulOp:$r
               (StableHLO_DivOp:$mn_over_mx $mn, $mx),
               $mn_over_mx))),
         $constant_1,
          StableHLO_ComparisonDirectionValue<"EQ">,
          (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
        (StableHLO_CompareOp
         $r,
         (StableHLO_ConstantLike<"0"> $x),
          StableHLO_ComparisonDirectionValue<"GT">,
          (STABLEHLO_DEFAULT_COMPARISON_TYPE))),
      (StableHLO_AddOp
        $mx,
        (StableHLO_DivOp
          (StableHLO_MulOp $mx, $r),
          $constant_2)),
      (StableHLO_MulOp $mx, $sqa)))>;