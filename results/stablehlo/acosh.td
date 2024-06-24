// This file is generated using functional_algorithms tool (0.4.0), see
//   https://github.com/pearu/functional_algorithms
// for more information.




def : Pat<(acosh_0 NonComplexElementType:$z),
  (StableHLO_SelectOp
    (StableHLO_CompareOp
     $z,
     (StableHLO_DivOp
       (StableHLO_ConstantLikeMaxFiniteValue $z),
       (StableHLO_ConstantLike<"2">:$constant_2 $z)),
      StableHLO_ComparisonDirectionValue<"GE">,
      (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
    (StableHLO_AddOp
      (StableHLO_LogOp $constant_2),
      (StableHLO_LogOp $z)),
    (StableHLO_Log1pOp
      (StableHLO_MulOp
        (StableHLO_SqrtOp:$sqrt_subtract_z_constant_1
          (StableHLO_SubtractOp
            $z,
            (StableHLO_ConstantLike<"1">:$constant_1 $z))),
        (StableHLO_AddOp
          (StableHLO_SqrtOp
            (StableHLO_AddOp $z, $constant_1)),
          $sqrt_subtract_z_constant_1))))>;

def : Pat<(acosh_1 ComplexElementType:$z),
  (StableHLO_SelectOp
    (StableHLO_CompareOp
     (StableHLO_ImagOp:$signed_y $z),
     (StableHLO_ConstantLike<"0"> $signed_y),
      StableHLO_ComparisonDirectionValue<"LT">,
      (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
    (StableHLO_NegOp
      (StableHLO_ComplexOp:$complex_negative_acos_signed_imag_acos_real
        (StableHLO_NegOp
          (StableHLO_SelectOp
            (StableHLO_CompareOp
             $signed_y,
             (StableHLO_ConstantLike<"0">:$zero (StableHLO_RealOp:$signed_x $z)),
              StableHLO_ComparisonDirectionValue<"LT">,
              (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
            (StableHLO_SelectOp:$imag
              (StableHLO_CompareOp
               (StableHLO_SelectOp:$mx
                 (StableHLO_CompareOp:$y_gt_safe_max_opt
                  (StableHLO_AbsOp:$y $signed_y),
                  (StableHLO_SelectOp:$safe_max_opt
                    (StableHLO_CompareOp
                     (StableHLO_AbsOp:$x $signed_x),
                     (StableHLO_MulOp
                       (StableHLO_DivOp:$safe_max
                         (StableHLO_SqrtOp
                           (StableHLO_ConstantLikeMaxFiniteValue $signed_x)),
                         (StableHLO_ConstantLike<"8"> $signed_x)),
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
                  (StableHLO_LogOp
                    (StableHLO_ConstantLike<"2">:$two $signed_x)),
                  (StableHLO_LogOp $mx)),
                (StableHLO_MulOp
                  (StableHLO_ConstantLike<"0.5">:$half $signed_x),
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
                     (StableHLO_ConstantLike<"4"> $signed_x)),
                    StableHLO_ComparisonDirectionValue<"LT">,
                    (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
                  (StableHLO_CompareOp
                   $x,
                   (StableHLO_ConstantLike<"1">:$one $signed_x),
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
                        (StableHLO_SelectOp:$x_ge_1_or_not
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
                             (StableHLO_ConstantLike<"1.5"> $signed_x),
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
                  (StableHLO_AddOp $am1, $sq)))),
            (StableHLO_NegOp $imag))),
        (StableHLO_Atan2Op:$acos_real
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
                    (StableHLO_DivOp $half_apx, $spxm1)))))),
          $signed_x))),
    $complex_negative_acos_signed_imag_acos_real)>;