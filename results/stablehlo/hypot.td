// This file is generated using functional_algorithms tool (0.1.dev6+g8b58236), see
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
      $mx,
      (StableHLO_SqrtOp
        (StableHLO_ConstantLike<"2"> $x))),
    (StableHLO_MulOp
      $mx,
      (StableHLO_SqrtOp
        (StableHLO_AddOp
          (StableHLO_MulOp
            (StableHLO_DivOp:$z $mn, $mx),
            $z),
          (StableHLO_ConstantLike<"1"> $x)))))>;