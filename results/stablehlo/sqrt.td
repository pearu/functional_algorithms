// This file is generated using functional_algorithms tool (0.11.1), see
//   https://github.com/pearu/functional_algorithms
// for more information.




def : Pat<(sqrt_0 ComplexElementType:$z),
  (StableHLO_ComplexOp
    (StableHLO_SelectOp
      (StableHLO_CompareOp
       (StableHLO_RealOp:$x $z),
       (StableHLO_ConstantLike<"0.0">:$constant_f0 $x),
        StableHLO_ComparisonDirectionValue<"GE">,
        (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
      (StableHLO_SelectOp:$u
        (StableHLO_CompareOp:$eq_ax_ay
         (StableHLO_AbsOp:$ax $x),
         (StableHLO_AbsOp:$ay
           (StableHLO_ImagOp:$y $z)),
          StableHLO_ComparisonDirectionValue<"EQ">,
          (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
        (StableHLO_DivOp
          (StableHLO_MulOp
            (StableHLO_SqrtOp:$sq_ax $ax),
            (StableHLO_ConstantLike<"1.5537739740300374"> $x)),
          (StableHLO_ConstantLike<"1.4142135623730951">:$sq_2 $x)),
        (StableHLO_SelectOp
          (StableHLO_OrOp:$logical_or_eq_u_general_constant_f0_eq_u_general_constant_posinf
            (StableHLO_CompareOp
             (StableHLO_SqrtOp:$u_general
               (StableHLO_AddOp
                 (StableHLO_DivOp
                   (StableHLO_SelectOp
                     (StableHLO_CompareOp
                      (StableHLO_MaxOp:$mx $ax, $ay),
                      (StableHLO_MinOp:$mn $ax, $ay),
                       StableHLO_ComparisonDirectionValue<"EQ">,
                       (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
                     (StableHLO_MulOp $sq_2, $mx),
                     (StableHLO_SelectOp
                       (StableHLO_AndOp
                         (StableHLO_CompareOp
                          (StableHLO_SqrtOp:$sqa
                            (StableHLO_AddOp
                              (StableHLO_ConstantLike<"1.0">:$one $x),
                              (StableHLO_MulOp:$r
                                (StableHLO_DivOp:$mn_over_mx $mn, $mx),
                                $mn_over_mx))),
                          $one,
                           StableHLO_ComparisonDirectionValue<"EQ">,
                           (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
                         (StableHLO_CompareOp
                          $r,
                          $constant_f0,
                           StableHLO_ComparisonDirectionValue<"GT">,
                           (STABLEHLO_DEFAULT_COMPARISON_TYPE))),
                       (StableHLO_AddOp
                         $mx,
                         (StableHLO_DivOp
                           (StableHLO_MulOp $mx, $r),
                           (StableHLO_ConstantLike<"2.0">:$two $x))),
                       (StableHLO_MulOp $mx, $sqa))),
                   $two),
                 (StableHLO_DivOp $ax, $two))),
             $constant_f0,
              StableHLO_ComparisonDirectionValue<"EQ">,
              (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
            (StableHLO_CompareOp
             $u_general,
             (StableHLO_ConstantLikePosInfValue $x),
              StableHLO_ComparisonDirectionValue<"EQ">,
              (STABLEHLO_DEFAULT_COMPARISON_TYPE))),
          (StableHLO_SelectOp
            (StableHLO_CompareOp:$gt_ax_ay
             $ax,
             $ay,
              StableHLO_ComparisonDirectionValue<"GT">,
              (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
            (StableHLO_MulOp
              $sq_ax,
              (StableHLO_DivOp
                (StableHLO_SqrtOp:$sq_1h
                  (StableHLO_AddOp
                    $one,
                    (StableHLO_SelectOp:$h
                      (StableHLO_CompareOp
                       (StableHLO_MaxOp:$_mx_0_
                         $one,
                         (StableHLO_AbsOp:$abs__r_0_
                           (StableHLO_SelectOp:$_r_0_
                             $eq_ax_ay,
                             $one,
                             (StableHLO_SelectOp
                               (StableHLO_CompareOp:$lt_ax_ay
                                $ax,
                                $ay,
                                 StableHLO_ComparisonDirectionValue<"LT">,
                                 (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
                               (StableHLO_DivOp $ax, $ay),
                               (StableHLO_DivOp $ay, $ax))))),
                       (StableHLO_MinOp:$_mn_0_ $one, $abs__r_0_),
                        StableHLO_ComparisonDirectionValue<"EQ">,
                        (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
                      (StableHLO_MulOp $sq_2, $_mx_0_),
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
                           $constant_f0,
                            StableHLO_ComparisonDirectionValue<"GT">,
                            (STABLEHLO_DEFAULT_COMPARISON_TYPE))),
                        (StableHLO_AddOp
                          $_mx_0_,
                          (StableHLO_DivOp
                            (StableHLO_MulOp $_mx_0_, $_r_1_),
                            $two)),
                        (StableHLO_MulOp $_mx_0_, $_sqa_0_))))),
                $sq_2)),
            (StableHLO_MulOp
              (StableHLO_SqrtOp:$sq_ay $ay),
              (StableHLO_DivOp
                (StableHLO_SqrtOp:$sq_rh
                  (StableHLO_AddOp $_r_0_, $h)),
                $sq_2))),
          $u_general)),
      (StableHLO_SelectOp:$ay_div_u
        $eq_ax_ay,
        (StableHLO_DivOp
          $sq_ay,
          (StableHLO_ConstantLike<"2.19736822693562"> $x)),
        (StableHLO_SelectOp
          $logical_or_eq_u_general_constant_f0_eq_u_general_constant_posinf,
          (StableHLO_SelectOp
            $gt_ax_ay,
            (StableHLO_DivOp
              (StableHLO_MulOp
                $sq_ay,
                (StableHLO_SelectOp
                  $eq_ax_ay,
                  $one,
                  (StableHLO_SelectOp
                    $lt_ax_ay,
                    (StableHLO_DivOp $sq_ax, $sq_ay),
                    (StableHLO_DivOp $sq_ay, $sq_ax)))),
              (StableHLO_MulOp $sq_1h, $sq_2)),
            (StableHLO_DivOp
              $sq_ay,
              (StableHLO_MulOp $sq_rh, $sq_2))),
          (StableHLO_DivOp
            $ay,
            (StableHLO_MulOp $u_general, $two))))),
    (StableHLO_SelectOp
      (StableHLO_CompareOp
       $x,
       $constant_f0,
        StableHLO_ComparisonDirectionValue<"LT">,
        (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
      (StableHLO_SelectOp
        (StableHLO_CompareOp:$lt_y_constant_f0
         $y,
         $constant_f0,
          StableHLO_ComparisonDirectionValue<"LT">,
          (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
        (StableHLO_NegOp $u),
        $u),
      (StableHLO_SelectOp
        $lt_y_constant_f0,
        (StableHLO_NegOp $ay_div_u),
        $ay_div_u)))>;