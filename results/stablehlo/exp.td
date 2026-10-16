// This file is generated using functional_algorithms tool (0.15.1.dev20+g7dbbfd7db.d20261016), see
//   https://github.com/pearu/functional_algorithms
// for more information.




def : Pat<(exp_0 ComplexElementType:$z),
  (StableHLO_ComplexOp
    (StableHLO_SelectOp
      (StableHLO_AndOp
        (StableHLO_NotOp:$y_is_not_finite
          (StableHLO_IsFiniteOp
            (StableHLO_ImagOp:$y $z))),
        (StableHLO_OrOp
          (StableHLO_CompareOp
           (StableHLO_RealOp:$x $z),
           (StableHLO_ConstantLikePosInfValue:$constant_posinf $x),
            StableHLO_ComparisonDirectionValue<"EQ">,
            (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
          (StableHLO_CompareOp:$eq_x_constant_neginf
           $x,
           (StableHLO_ConstantLikeNegInfValue $x),
            StableHLO_ComparisonDirectionValue<"EQ">,
            (STABLEHLO_DEFAULT_COMPARISON_TYPE)))),
      (StableHLO_ExpOp:$e $x),
      (StableHLO_SelectOp
        (StableHLO_CompareOp:$eq_e_constant_posinf
         $e,
         $constant_posinf,
          StableHLO_ComparisonDirectionValue<"EQ">,
          (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
        (StableHLO_MulOp
          (StableHLO_MulOp
            (StableHLO_ExpOp:$e2
              (StableHLO_MulOp
                $x,
                (StableHLO_ConstantLike<"0.5"> $x))),
            (StableHLO_CosineOp:$cs $y)),
          $e2),
        (StableHLO_MulOp $e, $cs))),
    (StableHLO_SelectOp
      (StableHLO_AndOp
        (StableHLO_NotOp
          (StableHLO_AndOp $eq_x_constant_neginf, $y_is_not_finite)),
        (StableHLO_CompareOp
         $y,
         (StableHLO_ConstantLike<"0.0">:$zero $x),
          StableHLO_ComparisonDirectionValue<"NE">,
          (STABLEHLO_DEFAULT_COMPARISON_TYPE))),
      (StableHLO_SelectOp
        $eq_e_constant_posinf,
        (StableHLO_MulOp
          (StableHLO_MulOp
            $e2,
            (StableHLO_SineOp:$sn $y)),
          $e2),
        (StableHLO_MulOp $e, $sn)),
      $zero))>;