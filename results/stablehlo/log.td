// This file is generated using functional_algorithms tool (0.14.1.dev0+ge22be68.d20241231), see
//   https://github.com/pearu/functional_algorithms
// for more information.




def : Pat<(log_0 ComplexElementType:$z),
  (StableHLO_ComplexOp
    (StableHLO_SelectOp
      (StableHLO_CompareOp
       (StableHLO_AbsOp
         (StableHLO_AddOp:$sum_fast2sum_high
           (StableHLO_AddOp:$add_fast2sum_high
             (StableHLO_AddOp:$_add_fast2sum_high_0_
               (StableHLO_AddOp:$_add_fast2sum_high_1_
                 (StableHLO_AddOp:$_add_fast2sum_high_2_
                   (StableHLO_ConstantLike<"-1.0">:$constant_fneg1 (StableHLO_RealOp:$x $z)),
                   (StableHLO_SelectOp:$mxh
                     (StableHLO_CompareOp:$gt_square_dekker_high__square_dekker_high_0_
                      (StableHLO_MulOp:$square_dekker_high
                        (StableHLO_ImagOp:$y $z),
                        $y),
                      (StableHLO_MulOp:$_square_dekker_high_0_ $x, $x),
                       StableHLO_ComparisonDirectionValue<"GT">,
                       (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
                     $square_dekker_high,
                     $_square_dekker_high_0_)),
                 (StableHLO_SelectOp:$mnh $gt_square_dekker_high__square_dekker_high_0_, $_square_dekker_high_0_, $square_dekker_high)),
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
                                (StableHLO_ConstantLikeMaxFiniteValue:$largest $x),
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
                 (StableHLO_SubtractOp:$add_fast2sum_low
                   $mxh,
                   (StableHLO_SubtractOp $_add_fast2sum_high_2_, $constant_fneg1)),
                 (StableHLO_SubtractOp:$_add_fast2sum_low_0_
                   $mnh,
                   (StableHLO_SubtractOp $_add_fast2sum_high_1_, $_add_fast2sum_high_2_))),
               (StableHLO_SubtractOp:$_add_fast2sum_low_1_
                 $square_dekker_low,
                 (StableHLO_SubtractOp $_add_fast2sum_high_0_, $_add_fast2sum_high_1_))),
             (StableHLO_SubtractOp:$_add_fast2sum_low_2_
               $_square_dekker_low_0_,
               (StableHLO_SubtractOp $add_fast2sum_high, $_add_fast2sum_high_0_))))),
       (StableHLO_ConstantLike<"0.5">:$half $x),
        StableHLO_ComparisonDirectionValue<"LT">,
        (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
      (StableHLO_MulOp
        $half,
        (StableHLO_Log1pOp $sum_fast2sum_high)),
      (StableHLO_AddOp
        (StableHLO_LogOp
          (StableHLO_MaxOp:$mx
            (StableHLO_AbsOp:$abs_x $x),
            (StableHLO_AbsOp:$abs_y $y))),
        (StableHLO_MulOp
          $half,
          (StableHLO_Log1pOp
            (StableHLO_MulOp
              (StableHLO_SelectOp:$mn_over_mx
                (StableHLO_CompareOp
                 (StableHLO_MinOp:$mn $abs_x, $abs_y),
                 $mx,
                  StableHLO_ComparisonDirectionValue<"EQ">,
                  (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
                (StableHLO_ConstantLike<"1.0"> $x),
                (StableHLO_DivOp $mn, $mx)),
              $mn_over_mx))))),
    (StableHLO_Atan2Op $y, $x))>;