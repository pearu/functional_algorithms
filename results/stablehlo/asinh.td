// This file is generated using functional_algorithms tool (0.4.0), see
//   https://github.com/pearu/functional_algorithms
// for more information.




def : Pat<(asinh_0 NonComplexElementType:$signed_y),
  (StableHLO_SelectOp
    (StableHLO_CompareOp
     $signed_y,
     (StableHLO_ConstantLike<"0">:$xoy $signed_y),
      StableHLO_ComparisonDirectionValue<"LT">,
      (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
    (StableHLO_NegOp
      (StableHLO_SelectOp:$imag
        (StableHLO_CompareOp
         (StableHLO_SelectOp:$mx
           (StableHLO_CompareOp:$y_gt_safe_max_opt
            (StableHLO_AbsOp:$y $signed_y),
            (StableHLO_MulOp:$safe_max_opt
              (StableHLO_DivOp:$safe_max
                (StableHLO_SqrtOp
                  (StableHLO_ConstantLikeMaxFiniteValue $signed_y)),
                (StableHLO_ConstantLike<"8"> $signed_y)),
              (StableHLO_ConstantLike<"1e-06"> $signed_y)),
             StableHLO_ComparisonDirectionValue<"GE">,
             (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
           $y,
           $xoy),
         (StableHLO_SelectOp $y_gt_safe_max_opt, $safe_max_opt, $safe_max),
          StableHLO_ComparisonDirectionValue<"GE">,
          (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
        (StableHLO_AddOp
          (StableHLO_LogOp
            (StableHLO_ConstantLike<"2">:$two $signed_y)),
          (StableHLO_LogOp $mx)),
        (StableHLO_SelectOp
          (StableHLO_CompareOp:$lt_y_safe_min
           $y,
           (StableHLO_MulOp
             (StableHLO_SqrtOp
               (StableHLO_ConstantLikeSmallestNormalizedValue $signed_y)),
             (StableHLO_ConstantLike<"4"> $signed_y)),
            StableHLO_ComparisonDirectionValue<"LT">,
            (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
          (StableHLO_DivOp
            $y,
            (StableHLO_SqrtOp:$sq
              (StableHLO_MulOp
                (StableHLO_SelectOp:$am1
                  $lt_y_safe_min,
                  (StableHLO_NegOp
                    (StableHLO_DivOp
                      (StableHLO_ConstantLike<"-1">:$xm1 $signed_y),
                      (StableHLO_AddOp:$ap1
                        (StableHLO_MulOp:$apx
                          (StableHLO_ConstantLike<"0.5">:$half $signed_y),
                          (StableHLO_AddOp
                            (StableHLO_SelectOp:$s
                              (StableHLO_CompareOp
                               (StableHLO_ConstantLike<"1">:$xp1 $signed_y),
                               $y,
                                StableHLO_ComparisonDirectionValue<"EQ">,
                                (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
                              (StableHLO_MulOp
                                (StableHLO_SqrtOp $two),
                                (StableHLO_MaxOp:$_hypot_2_mx $xp1, $y)),
                              (StableHLO_SelectOp
                                (StableHLO_AndOp
                                  (StableHLO_CompareOp
                                   (StableHLO_SqrtOp:$_hypot_2_sqa
                                     (StableHLO_AddOp
                                       $xp1,
                                       (StableHLO_MulOp:$_hypot_2_r
                                         (StableHLO_DivOp:$_hypot_2_mn_over_mx
                                           (StableHLO_MinOp $xp1, $y),
                                           $_hypot_2_mx),
                                         $_hypot_2_mn_over_mx))),
                                   $xp1,
                                    StableHLO_ComparisonDirectionValue<"EQ">,
                                    (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
                                  (StableHLO_CompareOp
                                   $_hypot_2_r,
                                   $xoy,
                                    StableHLO_ComparisonDirectionValue<"GT">,
                                    (STABLEHLO_DEFAULT_COMPARISON_TYPE))),
                                (StableHLO_AddOp
                                  $_hypot_2_mx,
                                  (StableHLO_DivOp
                                    (StableHLO_MulOp $_hypot_2_mx, $_hypot_2_r),
                                    $two)),
                                (StableHLO_MulOp $_hypot_2_mx, $_hypot_2_sqa))),
                            $s)),
                        $xp1))),
                  (StableHLO_SelectOp:$x_ge_1_or_not
                    (StableHLO_CompareOp
                     $apx,
                     (StableHLO_ConstantLike<"1.5"> $signed_y),
                      StableHLO_ComparisonDirectionValue<"LE">,
                      (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
                    (StableHLO_AddOp
                      (StableHLO_DivOp
                        (StableHLO_MulOp:$half_yy
                          $half,
                          (StableHLO_MulOp $y, $y)),
                        (StableHLO_AddOp $s, $xp1)),
                      (StableHLO_DivOp
                        $half_yy,
                        (StableHLO_SubtractOp $s, $xm1))),
                    (StableHLO_SubtractOp $apx, $xp1))),
                $ap1))),
          (StableHLO_Log1pOp
            (StableHLO_AddOp $am1, $sq))))),
    $imag)>;

def : Pat<(asinh_1 ComplexElementType:$z),
  (StableHLO_ComplexOp
    (StableHLO_SelectOp
      (StableHLO_CompareOp
       (StableHLO_RealOp:$signed_y $z),
       (StableHLO_ConstantLike<"0">:$zero (StableHLO_ImagOp:$imag_z $z)),
        StableHLO_ComparisonDirectionValue<"LT">,
        (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
      (StableHLO_NegOp
        (StableHLO_SelectOp:$imag
          (StableHLO_CompareOp
           (StableHLO_SelectOp:$mx
             (StableHLO_CompareOp:$y_gt_safe_max_opt
              (StableHLO_AbsOp:$y $signed_y),
              (StableHLO_SelectOp:$safe_max_opt
                (StableHLO_CompareOp
                 (StableHLO_AbsOp:$x
                   (StableHLO_NegOp:$signed_x $imag_z)),
                 (StableHLO_MulOp
                   (StableHLO_DivOp:$safe_max
                     (StableHLO_SqrtOp
                       (StableHLO_ConstantLikeMaxFiniteValue $imag_z)),
                     (StableHLO_ConstantLike<"8"> $imag_z)),
                   (StableHLO_ConstantLike<"1000000000000.0"> $imag_z)),
                  StableHLO_ComparisonDirectionValue<"LT">,
                  (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
                (StableHLO_MulOp
                  $safe_max,
                  (StableHLO_ConstantLike<"1e-06"> $imag_z)),
                (StableHLO_MulOp
                  $safe_max,
                  (StableHLO_ConstantLike<"100.0"> $imag_z))),
               StableHLO_ComparisonDirectionValue<"GE">,
               (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
             $y,
             $x),
           (StableHLO_SelectOp $y_gt_safe_max_opt, $safe_max_opt, $safe_max),
            StableHLO_ComparisonDirectionValue<"GE">,
            (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
          (StableHLO_AddOp
            (StableHLO_AddOp
              (StableHLO_LogOp
                (StableHLO_ConstantLike<"2">:$two $imag_z)),
              (StableHLO_LogOp $mx)),
            (StableHLO_MulOp
              (StableHLO_ConstantLike<"0.5">:$half $imag_z),
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
                   (StableHLO_ConstantLikeSmallestNormalizedValue $imag_z)),
                 (StableHLO_ConstantLike<"4"> $imag_z)),
                StableHLO_ComparisonDirectionValue<"LT">,
                (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
              (StableHLO_CompareOp
               $x,
               (StableHLO_ConstantLike<"1">:$one $imag_z),
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
                        (StableHLO_MulOp
                          (StableHLO_AddOp:$xp1 $x, $one),
                          (StableHLO_SubtractOp:$xm1 $x, $one)),
                        (StableHLO_AddOp:$ap1
                          (StableHLO_MulOp:$a
                            $half,
                            (StableHLO_AddOp
                              (StableHLO_SelectOp:$r
                                (StableHLO_CompareOp
                                 (StableHLO_MaxOp:$_hypot_1_mx
                                   (StableHLO_AbsOp:$abs_xp1 $xp1),
                                   $y),
                                 (StableHLO_MinOp:$mn $abs_xp1, $y),
                                  StableHLO_ComparisonDirectionValue<"EQ">,
                                  (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
                                (StableHLO_MulOp
                                  (StableHLO_SqrtOp:$sqrt_two $two),
                                  $_hypot_1_mx),
                                (StableHLO_SelectOp
                                  (StableHLO_AndOp
                                    (StableHLO_CompareOp
                                     (StableHLO_SqrtOp:$sqa
                                       (StableHLO_AddOp
                                         $one,
                                         (StableHLO_MulOp:$_hypot_1_r
                                           (StableHLO_DivOp:$mn_over_mx $mn, $_hypot_1_mx),
                                           $mn_over_mx))),
                                     $one,
                                      StableHLO_ComparisonDirectionValue<"EQ">,
                                      (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
                                    (StableHLO_CompareOp
                                     $_hypot_1_r,
                                     $zero,
                                      StableHLO_ComparisonDirectionValue<"GT">,
                                      (STABLEHLO_DEFAULT_COMPARISON_TYPE))),
                                  (StableHLO_AddOp
                                    $_hypot_1_mx,
                                    (StableHLO_DivOp
                                      (StableHLO_MulOp $_hypot_1_mx, $_hypot_1_r),
                                      $two)),
                                  (StableHLO_MulOp $_hypot_1_mx, $sqa))),
                              (StableHLO_SelectOp:$s
                                (StableHLO_CompareOp
                                 (StableHLO_MaxOp:$_hypot_2_mx
                                   (StableHLO_AbsOp:$abs_xm1 $xm1),
                                   $y),
                                 (StableHLO_MinOp:$_hypot_2_mn $abs_xm1, $y),
                                  StableHLO_ComparisonDirectionValue<"EQ">,
                                  (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
                                (StableHLO_MulOp $sqrt_two, $_hypot_2_mx),
                                (StableHLO_SelectOp
                                  (StableHLO_AndOp
                                    (StableHLO_CompareOp
                                     (StableHLO_SqrtOp:$_hypot_2_sqa
                                       (StableHLO_AddOp
                                         $one,
                                         (StableHLO_MulOp:$_hypot_2_r
                                           (StableHLO_DivOp:$_hypot_2_mn_over_mx $_hypot_2_mn, $_hypot_2_mx),
                                           $_hypot_2_mn_over_mx))),
                                     $one,
                                      StableHLO_ComparisonDirectionValue<"EQ">,
                                      (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
                                    (StableHLO_CompareOp
                                     $_hypot_2_r,
                                     $zero,
                                      StableHLO_ComparisonDirectionValue<"GT">,
                                      (STABLEHLO_DEFAULT_COMPARISON_TYPE))),
                                  (StableHLO_AddOp
                                    $_hypot_2_mx,
                                    (StableHLO_DivOp
                                      (StableHLO_MulOp $_hypot_2_mx, $_hypot_2_r),
                                      $two)),
                                  (StableHLO_MulOp $_hypot_2_mx, $_hypot_2_sqa))))),
                          $one))),
                    (StableHLO_SelectOp:$_asin_2_x_ge_1_or_not
                      (StableHLO_CompareOp
                       $x,
                       $one,
                        StableHLO_ComparisonDirectionValue<"GE">,
                        (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
                      (StableHLO_AddOp
                        (StableHLO_DivOp:$divide_half_yy_rpxp1
                          (StableHLO_MulOp:$half_yy
                            $half,
                            (StableHLO_MulOp:$yy $y, $y)),
                          (StableHLO_AddOp:$rpxp1 $r, $xp1)),
                        (StableHLO_MulOp
                          $half,
                          (StableHLO_AddOp:$spxm1 $s, $xm1))),
                      (StableHLO_SelectOp
                        (StableHLO_CompareOp
                         $a,
                         (StableHLO_ConstantLike<"1.5"> $imag_z),
                          StableHLO_ComparisonDirectionValue<"LE">,
                          (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
                        (StableHLO_AddOp
                          $divide_half_yy_rpxp1,
                          (StableHLO_DivOp
                            $half_yy,
                            (StableHLO_SubtractOp:$smxm1 $s, $xm1))),
                        (StableHLO_SubtractOp $a, $one)))),
                  $ap1))),
            (StableHLO_Log1pOp
              (StableHLO_AddOp $am1, $sq))))),
      $imag),
    (StableHLO_NegOp
      (StableHLO_Atan2Op:$real
        $signed_x,
        (StableHLO_SelectOp
          (StableHLO_CompareOp
           (StableHLO_MaxOp $x, $y),
           $safe_max,
            StableHLO_ComparisonDirectionValue<"GE">,
            (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
          $y,
          (StableHLO_SelectOp
            (StableHLO_CompareOp
             $x,
             $one,
              StableHLO_ComparisonDirectionValue<"LE">,
              (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
            (StableHLO_SqrtOp
              (StableHLO_MulOp
                (StableHLO_MulOp:$half_apx
                  $half,
                  (StableHLO_AddOp $a, $x)),
                (StableHLO_AddOp
                  (StableHLO_DivOp $yy, $rpxp1),
                  $smxm1))),
            (StableHLO_MulOp
              $y,
              (StableHLO_SqrtOp
                (StableHLO_AddOp
                  (StableHLO_DivOp $half_apx, $rpxp1),
                  (StableHLO_DivOp $half_apx, $spxm1)))))))))>;