// This file is generated using functional_algorithms tool (0.11.1), see
//   https://github.com/pearu/functional_algorithms
// for more information.




def : Pat<(log1p_1 ComplexElementType:$z),
  (StableHLO_ComplexOp
    (StableHLO_SelectOp
      (StableHLO_CompareOp
       (StableHLO_MaxOp:$mx
         (StableHLO_AbsOp:$ax
           (StableHLO_RealOp:$x $z)),
         (StableHLO_AbsOp:$ay
           (StableHLO_ImagOp:$y $z))),
       (StableHLO_MulOp
         (StableHLO_SqrtOp
           (StableHLO_ConstantLikeMaxFiniteValue:$largest $x)),
         (StableHLO_ConstantLike<"0.01"> $x)),
        StableHLO_ComparisonDirectionValue<"GT">,
        (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
      (StableHLO_AddOp
        (StableHLO_LogOp $mx),
        (StableHLO_MulOp
          (StableHLO_ConstantLike<"0.5">:$half $x),
          (StableHLO_Log1pOp
            (StableHLO_SelectOp
              (StableHLO_CompareOp
               (StableHLO_MinOp:$mn $ax, $ay),
               $mx,
                StableHLO_ComparisonDirectionValue<"EQ">,
                (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
              (StableHLO_ConstantLike<"1.0">:$one $x),
              (StableHLO_MulOp
                (StableHLO_DivOp:$r $mn, $mx),
                $r))))),
      (StableHLO_SelectOp
        (StableHLO_CompareOp
         (StableHLO_AddOp
           (StableHLO_AbsOp
             (StableHLO_AddOp:$xp1 $x, $one)),
           $ay),
         (StableHLO_ConstantLike<"0.2"> $x),
          StableHLO_ComparisonDirectionValue<"LT">,
          (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
        (StableHLO_MulOp
          $half,
          (StableHLO_LogOp
            (StableHLO_AddOp
              (StableHLO_MulOp $xp1, $xp1),
              (StableHLO_MulOp:$square_dekker_high $y, $y)))),
        (StableHLO_MulOp
          $half,
          (StableHLO_Log1pOp
            (StableHLO_AddOp:$sum_2sum_high
              (StableHLO_AddOp:$add_2sum_high
                (StableHLO_AddOp:$_add_2sum_high_0_
                  (StableHLO_AddOp:$_add_2sum_high_1_
                    (StableHLO_AddOp:$_add_2sum_high_2_
                      (StableHLO_AddOp:$x2h $x, $x),
                      $square_dekker_high),
                    (StableHLO_MulOp:$_square_dekker_high_0_ $x, $x)),
                  (StableHLO_AddOp:$square_dekker_low
                    (StableHLO_AddOp
                      (StableHLO_AddOp
                        (StableHLO_AddOp
                          (StableHLO_NegOp $square_dekker_high),
                          (StableHLO_MulOp
                            (StableHLO_AddOp:$yh
                              (StableHLO_MulOp:$multiply_veltkamp_splitter_constant_y
                                (StableHLO_SelectOp:$veltkamp_splitter_constant
                                  (StableHLO_CompareOp
                                   $largest,
                                   (StableHLO_ConstantLike<"1e+308"> $x),
                                    StableHLO_ComparisonDirectionValue<"GT">,
                                    (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
                                  (StableHLO_ConstantLike<"134217729.0"> $x),
                                  (StableHLO_SelectOp
                                    (StableHLO_CompareOp
                                     $largest,
                                     (StableHLO_ConstantLike<"1e+38"> $x),
                                      StableHLO_ComparisonDirectionValue<"GT">,
                                      (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
                                    (StableHLO_ConstantLike<"4097.0"> $x),
                                    (StableHLO_ConstantLike<"65.0"> $x))),
                                $y),
                              (StableHLO_SubtractOp $y, $multiply_veltkamp_splitter_constant_y)),
                            $yh)),
                        (StableHLO_MulOp:$multiply_yh_yl
                          $yh,
                          (StableHLO_SubtractOp:$yl $y, $yh))),
                      $multiply_yh_yl),
                    (StableHLO_MulOp $yl, $yl))),
                (StableHLO_AddOp:$_square_dekker_low_0_
                  (StableHLO_AddOp
                    (StableHLO_AddOp
                      (StableHLO_AddOp
                        (StableHLO_NegOp $_square_dekker_high_0_),
                        (StableHLO_MulOp
                          (StableHLO_AddOp:$xh
                            (StableHLO_MulOp:$multiply_veltkamp_splitter_constant_x $veltkamp_splitter_constant, $x),
                            (StableHLO_SubtractOp $x, $multiply_veltkamp_splitter_constant_x)),
                          $xh)),
                      (StableHLO_MulOp:$multiply_xh_xl
                        $xh,
                        (StableHLO_SubtractOp:$xl $x, $xh))),
                    $multiply_xh_xl),
                  (StableHLO_MulOp $xl, $xl))),
              (StableHLO_AddOp
                (StableHLO_AddOp
                  (StableHLO_AddOp
                    (StableHLO_AddOp:$add_2sum_low
                      (StableHLO_SubtractOp
                        $x2h,
                        (StableHLO_SubtractOp
                          $_add_2sum_high_2_,
                          (StableHLO_SubtractOp:$subtract__add_2sum_high_2__x2h $_add_2sum_high_2_, $x2h))),
                      (StableHLO_SubtractOp $square_dekker_high, $subtract__add_2sum_high_2__x2h)),
                    (StableHLO_AddOp:$_add_2sum_low_0_
                      (StableHLO_SubtractOp
                        $_add_2sum_high_2_,
                        (StableHLO_SubtractOp
                          $_add_2sum_high_1_,
                          (StableHLO_SubtractOp:$subtract__add_2sum_high_1___add_2sum_high_2_ $_add_2sum_high_1_, $_add_2sum_high_2_))),
                      (StableHLO_SubtractOp $_square_dekker_high_0_, $subtract__add_2sum_high_1___add_2sum_high_2_))),
                  (StableHLO_AddOp:$_add_2sum_low_1_
                    (StableHLO_SubtractOp
                      $_add_2sum_high_1_,
                      (StableHLO_SubtractOp
                        $_add_2sum_high_0_,
                        (StableHLO_SubtractOp:$subtract__add_2sum_high_0___add_2sum_high_1_ $_add_2sum_high_0_, $_add_2sum_high_1_))),
                    (StableHLO_SubtractOp $square_dekker_low, $subtract__add_2sum_high_0___add_2sum_high_1_))),
                (StableHLO_AddOp:$_add_2sum_low_2_
                  (StableHLO_SubtractOp
                    $_add_2sum_high_0_,
                    (StableHLO_SubtractOp
                      $add_2sum_high,
                      (StableHLO_SubtractOp:$subtract_add_2sum_high__add_2sum_high_0_ $add_2sum_high, $_add_2sum_high_0_))),
                  (StableHLO_SubtractOp $_square_dekker_low_0_, $subtract_add_2sum_high__add_2sum_high_0_)))))))),
    (StableHLO_Atan2Op $y, $xp1))>;