// This file is generated using functional_algorithms tool (0.10.2.dev9+g7001467.d20241002), see
//   https://github.com/pearu/functional_algorithms
// for more information.




def : Pat<(atan_1 ComplexElementType:$z),
  (StableHLO_ComplexOp
    (StableHLO_MulOp
      (StableHLO_SelectOp
        (StableHLO_AndOp:$in_safe_region
          (StableHLO_CompareOp
           (StableHLO_AbsOp:$ax
             (StableHLO_NegOp:$x
               (StableHLO_ImagOp:$imag_z $z))),
           (StableHLO_MulOp:$safe_max
             (StableHLO_SelectOp:$inv_negeps
               (StableHLO_CompareOp
                (StableHLO_ConstantLikeMaxFiniteValue:$largest $imag_z),
                (StableHLO_ConstantLike<"1e+308"> $imag_z),
                 StableHLO_ComparisonDirectionValue<"GT">,
                 (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
               (StableHLO_ConstantLike<"9007199254740994.0"> $imag_z),
               (StableHLO_SelectOp
                 (StableHLO_CompareOp
                  $largest,
                  (StableHLO_ConstantLike<"1e+38"> $imag_z),
                   StableHLO_ComparisonDirectionValue<"GT">,
                   (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
                 (StableHLO_ConstantLike<"16777218.0"> $imag_z),
                 (StableHLO_ConstantLike<"2050.0"> $imag_z))),
             $inv_negeps),
            StableHLO_ComparisonDirectionValue<"LT">,
            (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
          (StableHLO_CompareOp
           (StableHLO_AbsOp:$ay
             (StableHLO_RealOp:$y $z)),
           $safe_max,
            StableHLO_ComparisonDirectionValue<"LT">,
            (STABLEHLO_DEFAULT_COMPARISON_TYPE))),
        (StableHLO_Atan2Op
          (StableHLO_AddOp $y, $y),
          (StableHLO_SubtractOp
            (StableHLO_MulOp
              (StableHLO_SubtractOp:$naxm1
                (StableHLO_ConstantLike<"1">:$one $imag_z),
                $ax),
              (StableHLO_AddOp $one, $ax)),
            (StableHLO_MulOp:$y2 $y, $y))),
        (StableHLO_MulOp
          (StableHLO_SelectOp
            (StableHLO_CompareOp
             $y,
             (StableHLO_ConstantLike<"0"> $y),
              StableHLO_ComparisonDirectionValue<"GE">,
              (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
            $one,
            (StableHLO_ConstantLike<"-1">:$constant_neg1 $imag_z)),
          (StableHLO_ConstantLike<"M_PI"> $imag_z))),
      (StableHLO_ConstantLike<"0.5"> $imag_z)),
    (StableHLO_NegOp
      (StableHLO_MulOp
        (StableHLO_MulOp
          (StableHLO_SelectOp
            (StableHLO_CompareOp
             $x,
             (StableHLO_ConstantLike<"0">:$zero $imag_z),
              StableHLO_ComparisonDirectionValue<"GE">,
              (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
            $one,
            $constant_neg1),
          (StableHLO_Log1pOp
            (StableHLO_MulOp
              (StableHLO_ConstantLike<"4"> $imag_z),
              (StableHLO_SelectOp
                $in_safe_region,
                (StableHLO_DivOp
                  $ax,
                  (StableHLO_AddOp
                    (StableHLO_MulOp $naxm1, $naxm1),
                    $y2)),
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
                         (StableHLO_ConstantLikePosInfValue:$constant_posinf $imag_z),
                          StableHLO_ComparisonDirectionValue<"EQ">,
                          (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
                        (StableHLO_CompareOp
                         $x,
                         (StableHLO_ConstantLikeNegInfValue:$constant_neginf $imag_z),
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
        (StableHLO_ConstantLike<"0.25"> $imag_z))))>;