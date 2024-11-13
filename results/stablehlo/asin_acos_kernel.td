// This file is generated using functional_algorithms tool (0.11.0), see
//   https://github.com/pearu/functional_algorithms
// for more information.




def : Pat<(asin_acos_kernel_0 ComplexElementType:$z),
  (StableHLO_ComplexOp
    (StableHLO_SelectOp
      (StableHLO_CompareOp
       (StableHLO_MaxOp
         (StableHLO_AbsOp:$x
           (StableHLO_RealOp:$signed_x $z)),
         (StableHLO_AbsOp:$y
           (StableHLO_ImagOp:$signed_y $z))),
       (StableHLO_DivOp:$safe_max
         (StableHLO_SqrtOp
           (StableHLO_ConstantLikeMaxFiniteValue $signed_x)),
         (StableHLO_ConstantLike<"8.0"> $signed_x)),
        StableHLO_ComparisonDirectionValue<"GE">,
        (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
      $y,
      (StableHLO_SelectOp
        (StableHLO_CompareOp
         $x,
         (StableHLO_ConstantLike<"1.0">:$one $signed_x),
          StableHLO_ComparisonDirectionValue<"LE">,
          (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
        (StableHLO_SqrtOp
          (StableHLO_MulOp
            (StableHLO_MulOp:$half_apx
              (StableHLO_ConstantLike<"0.5">:$half $signed_x),
              (StableHLO_AddOp
                (StableHLO_MulOp:$a
                  $half,
                  (StableHLO_AddOp
                    (StableHLO_SelectOp:$r
                      (StableHLO_CompareOp
                       (StableHLO_MaxOp:$mx
                         (StableHLO_AbsOp:$abs_xp1
                           (StableHLO_AddOp:$xp1 $x, $one)),
                         $y),
                       (StableHLO_MinOp:$mn $abs_xp1, $y),
                        StableHLO_ComparisonDirectionValue<"EQ">,
                        (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
                      (StableHLO_MulOp
                        (StableHLO_ConstantLike<"1.4142135623730951">:$sqrt_two $signed_x),
                        $mx),
                      (StableHLO_SelectOp
                        (StableHLO_AndOp
                          (StableHLO_CompareOp
                           (StableHLO_SqrtOp:$sqa
                             (StableHLO_AddOp
                               $one,
                               (StableHLO_MulOp:$_r_0_
                                 (StableHLO_DivOp:$mn_over_mx $mn, $mx),
                                 $mn_over_mx))),
                           $one,
                            StableHLO_ComparisonDirectionValue<"EQ">,
                            (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
                          (StableHLO_CompareOp
                           $_r_0_,
                           (StableHLO_ConstantLike<"0.0">:$zero $signed_x),
                            StableHLO_ComparisonDirectionValue<"GT">,
                            (STABLEHLO_DEFAULT_COMPARISON_TYPE))),
                        (StableHLO_AddOp
                          $mx,
                          (StableHLO_DivOp
                            (StableHLO_MulOp $mx, $_r_0_),
                            (StableHLO_ConstantLike<"2.0">:$two $signed_x))),
                        (StableHLO_MulOp $mx, $sqa))),
                    (StableHLO_SelectOp:$s
                      (StableHLO_CompareOp
                       (StableHLO_MaxOp:$_mx_0_
                         (StableHLO_AbsOp:$abs_xm1
                           (StableHLO_SubtractOp:$xm1 $x, $one)),
                         $y),
                       (StableHLO_MinOp:$_mn_0_ $abs_xm1, $y),
                        StableHLO_ComparisonDirectionValue<"EQ">,
                        (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
                      (StableHLO_MulOp $sqrt_two, $_mx_0_),
                      (StableHLO_SelectOp
                        (StableHLO_AndOp
                          (StableHLO_CompareOp
                           (StableHLO_SqrtOp:$_sqa_0_
                             (StableHLO_AddOp
                               $one,
                               (StableHLO_MulOp:$_r_1_
                                 (StableHLO_DivOp:$_mn_over_mx_0_ $_mn_0_, $_mx_0_),
                                 $_mn_over_mx_0_))),
                           $one,
                            StableHLO_ComparisonDirectionValue<"EQ">,
                            (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
                          (StableHLO_CompareOp
                           $_r_1_,
                           $zero,
                            StableHLO_ComparisonDirectionValue<"GT">,
                            (STABLEHLO_DEFAULT_COMPARISON_TYPE))),
                        (StableHLO_AddOp
                          $_mx_0_,
                          (StableHLO_DivOp
                            (StableHLO_MulOp $_mx_0_, $_r_1_),
                            $two)),
                        (StableHLO_MulOp $_mx_0_, $_sqa_0_))))),
                $x)),
            (StableHLO_AddOp
              (StableHLO_DivOp
                (StableHLO_MulOp:$yy $y, $y),
                (StableHLO_AddOp:$rpxp1 $r, $xp1)),
              (StableHLO_SubtractOp:$smxm1 $s, $xm1)))),
        (StableHLO_MulOp
          $y,
          (StableHLO_SqrtOp
            (StableHLO_AddOp
              (StableHLO_DivOp $half_apx, $rpxp1),
              (StableHLO_DivOp
                $half_apx,
                (StableHLO_AddOp:$spxm1 $s, $xm1))))))),
    (StableHLO_SelectOp
      (StableHLO_CompareOp
       (StableHLO_SelectOp:$_mx_1_
         (StableHLO_CompareOp:$y_gt_safe_max_opt
          $y,
          (StableHLO_SelectOp:$safe_max_opt
            (StableHLO_CompareOp
             $x,
             (StableHLO_MulOp
               $safe_max,
               (StableHLO_ConstantLike<"1000000000000.0"> $signed_x)),
              StableHLO_ComparisonDirectionValue<"LT">,
              (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
            (StableHLO_MulOp
              $safe_max,
              (StableHLO_ConstantLike<"1e-06"> $signed_x)),
            (StableHLO_MulOp
              $safe_max,
              (StableHLO_ConstantLike<"100.0"> $signed_x))),
           StableHLO_ComparisonDirectionValue<"GE">,
           (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
         $y,
         $x),
       (StableHLO_SelectOp $y_gt_safe_max_opt, $safe_max_opt, $safe_max),
        StableHLO_ComparisonDirectionValue<"GE">,
        (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
      (StableHLO_AddOp
        (StableHLO_AddOp
          (StableHLO_LogOp $two),
          (StableHLO_LogOp $_mx_1_)),
        (StableHLO_MulOp
          $half,
          (StableHLO_Log1pOp
            (StableHLO_MulOp
              (StableHLO_SelectOp:$xoy
                (StableHLO_AndOp
                  $y_gt_safe_max_opt,
                  (StableHLO_NotOp
                    (StableHLO_CompareOp
                     $y,
                     (StableHLO_ConstantLikePosInfValue $signed_y),
                      StableHLO_ComparisonDirectionValue<"EQ">,
                      (STABLEHLO_DEFAULT_COMPARISON_TYPE)))),
                (StableHLO_DivOp $x, $y),
                $zero),
              $xoy)))),
      (StableHLO_SelectOp
        (StableHLO_AndOp:$logical_and_lt_y_safe_min_lt_x_one
          (StableHLO_CompareOp
           $y,
           (StableHLO_MulOp
             (StableHLO_SqrtOp
               (StableHLO_ConstantLikeSmallestNormalizedValue $signed_x)),
             (StableHLO_ConstantLike<"4.0"> $signed_x)),
            StableHLO_ComparisonDirectionValue<"LT">,
            (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
          (StableHLO_CompareOp
           $x,
           $one,
            StableHLO_ComparisonDirectionValue<"LT">,
            (STABLEHLO_DEFAULT_COMPARISON_TYPE))),
        (StableHLO_DivOp
          $y,
          (StableHLO_SqrtOp:$sq
            (StableHLO_MulOp
              (StableHLO_SelectOp:$am1
                $logical_and_lt_y_safe_min_lt_x_one,
                (StableHLO_NegOp
                  (StableHLO_DivOp
                    (StableHLO_MulOp $xp1, $xm1),
                    (StableHLO_AddOp:$ap1 $a, $one))),
                (StableHLO_SelectOp:$x_ge_1_or_not
                  (StableHLO_CompareOp
                   $x,
                   $one,
                    StableHLO_ComparisonDirectionValue<"GE">,
                    (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
                  (StableHLO_AddOp
                    (StableHLO_DivOp:$divide_half_yy_rpxp1
                      (StableHLO_MulOp:$half_yy $half, $yy),
                      $rpxp1),
                    (StableHLO_MulOp $half, $spxm1)),
                  (StableHLO_SelectOp
                    (StableHLO_CompareOp
                     $a,
                     (StableHLO_ConstantLike<"1.5"> $signed_x),
                      StableHLO_ComparisonDirectionValue<"LE">,
                      (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
                    (StableHLO_AddOp
                      $divide_half_yy_rpxp1,
                      (StableHLO_DivOp $half_yy, $smxm1)),
                    (StableHLO_SubtractOp $a, $one)))),
              $ap1))),
        (StableHLO_Log1pOp
          (StableHLO_AddOp $am1, $sq)))))>;