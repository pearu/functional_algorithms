// This file is generated using functional_algorithms tool (0.10.2.dev9+g7001467.d20241002), see
//   https://github.com/pearu/functional_algorithms
// for more information.




def : Pat<(log1p_1 ComplexElementType:$z),
  (StableHLO_ComplexOp
    (StableHLO_AddOp
      (StableHLO_Log1pOp
        (StableHLO_SelectOp
          (StableHLO_CompareOp
           (StableHLO_AddOp:$xp1
             (StableHLO_RealOp:$x $z),
             (StableHLO_ConstantLike<"1.0">:$one $x)),
           (StableHLO_AbsOp:$ay
             (StableHLO_ImagOp:$y $z)),
            StableHLO_ComparisonDirectionValue<"GE">,
            (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
          $x,
          (StableHLO_SubtractOp
            (StableHLO_MaxOp:$mx
              (StableHLO_AbsOp:$axp1 $xp1),
              $ay),
            $one))),
      (StableHLO_MulOp
        (StableHLO_ConstantLike<"0.5"> $x),
        (StableHLO_Log1pOp
          (StableHLO_SelectOp
            (StableHLO_CompareOp
             (StableHLO_MinOp:$mn $axp1, $ay),
             $mx,
              StableHLO_ComparisonDirectionValue<"EQ">,
              (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
            $one,
            (StableHLO_MulOp
              (StableHLO_DivOp:$r $mn, $mx),
              $r))))),
    (StableHLO_Atan2Op $y, $xp1))>;