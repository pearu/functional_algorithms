// This file is generated using functional_algorithms tool (0.10.2.dev9+g7001467.d20241002), see
//   https://github.com/pearu/functional_algorithms
// for more information.




def : Pat<(atanh_1 ComplexElementType:$z),
  (StableHLO_ComplexOp
    (StableHLO_MulOp
      (StableHLO_MulOp
        (StableHLO_SelectOp
          (StableHLO_CompareOp
           (StableHLO_RealOp:$x $z),
           (StableHLO_ConstantLike<"0.0">:$zero $x),
            StableHLO_ComparisonDirectionValue<"GE">,
            (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
          (StableHLO_ConstantLike<"1.0">:$one $x),
          (StableHLO_ConstantLike<"-1.0">:$constant_fneg1 $x)),
        (StableHLO_Log1pOp
          (StableHLO_MulOp
            (StableHLO_ConstantLike<"4.0"> $x),
            (StableHLO_SelectOp
              (StableHLO_AndOp:$in_safe_region
                (StableHLO_CompareOp
                 (StableHLO_AbsOp:$ax $x),
                 (StableHLO_MulOp:$safe_max
                   (StableHLO_SelectOp:$inv_negeps
                     (StableHLO_CompareOp
                      (StableHLO_ConstantLikeMaxFiniteValue:$largest $x),
                      (StableHLO_ConstantLike<"1e+308"> $x),
                       StableHLO_ComparisonDirectionValue<"GT">,
                       (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
                     (StableHLO_ConstantLike<"9007199254740994.0"> $x),
                     (StableHLO_SelectOp
                       (StableHLO_CompareOp
                        $largest,
                        (StableHLO_ConstantLike<"1e+38"> $x),
                         StableHLO_ComparisonDirectionValue<"GT">,
                         (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
                       (StableHLO_ConstantLike<"16777218.0"> $x),
                       (StableHLO_ConstantLike<"2050.0"> $x))),
                   $inv_negeps),
                  StableHLO_ComparisonDirectionValue<"LT">,
                  (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
                (StableHLO_CompareOp
                 (StableHLO_AbsOp:$ay
                   (StableHLO_ImagOp:$y $z)),
                 $safe_max,
                  StableHLO_ComparisonDirectionValue<"LT">,
                  (STABLEHLO_DEFAULT_COMPARISON_TYPE))),
              (StableHLO_DivOp
                $ax,
                (StableHLO_AddOp
                  (StableHLO_MulOp
                    (StableHLO_SubtractOp:$naxm1 $one, $ax),
                    $naxm1),
                  (StableHLO_MulOp:$y2 $y, $y))),
              (StableHLO_SelectOp
                (StableHLO_CompareOp
                 (StableHLO_MulOp $ay, $inv_negeps),
                 $ax,
                  StableHLO_ComparisonDirectionValue<"LT">,
                  (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
                (StableHLO_DivOp $one, $ax),
                (StableHLO_SelectOp
                  (StableHLO_OrOp
                    (StableHLO_OrOp
                      (StableHLO_CompareOp
                       $x,
                       (StableHLO_ConstantLikePosInfValue:$constant_posinf $x),
                        StableHLO_ComparisonDirectionValue<"EQ">,
                        (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
                      (StableHLO_CompareOp
                       $x,
                       (StableHLO_ConstantLikeNegInfValue:$constant_neginf $x),
                        StableHLO_ComparisonDirectionValue<"EQ">,
                        (STABLEHLO_DEFAULT_COMPARISON_TYPE))),
                    (StableHLO_OrOp
                      (StableHLO_CompareOp
                       $y,
                       $constant_posinf,
                        StableHLO_ComparisonDirectionValue<"EQ">,
                        (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
                      (StableHLO_CompareOp
                       $y,
                       $constant_neginf,
                        StableHLO_ComparisonDirectionValue<"EQ">,
                        (STABLEHLO_DEFAULT_COMPARISON_TYPE)))),
                  $zero,
                  (StableHLO_DivOp
                    (StableHLO_DivOp
                      $one,
                      (StableHLO_AddOp
                        (StableHLO_DivOp $ax, $y),
                        (StableHLO_DivOp $y, $ax))),
                    $y))))))),
      (StableHLO_ConstantLike<"0.25"> $x)),
    (StableHLO_MulOp
      (StableHLO_SelectOp
        $in_safe_region,
        (StableHLO_Atan2Op
          (StableHLO_AddOp $y, $y),
          (StableHLO_SubtractOp
            (StableHLO_MulOp
              $naxm1,
              (StableHLO_AddOp $one, $ax)),
            $y2)),
        (StableHLO_MulOp
          (StableHLO_SelectOp
            (StableHLO_CompareOp
             $y,
             (StableHLO_ConstantLike<"0.0"> $y),
              StableHLO_ComparisonDirectionValue<"GE">,
              (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
            $one,
            $constant_fneg1),
          (StableHLO_ConstantLike<"M_PI"> $x))),
      (StableHLO_ConstantLike<"0.5"> $x)))>;