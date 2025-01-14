// This file is generated using functional_algorithms tool (0.15.1.dev3+ge93b47e.d20250113), see
//   https://github.com/pearu/functional_algorithms
// for more information.




def : Pat<(exp_0 ComplexElementType:$z),
  (StableHLO_ComplexOp
    (StableHLO_SelectOp
      (StableHLO_CompareOp:$eq_e_constant_posinf
       (StableHLO_ExpOp:$e
         (StableHLO_RealOp:$x $z)),
       (StableHLO_ConstantLikePosInfValue $x),
        StableHLO_ComparisonDirectionValue<"EQ">,
        (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
      (StableHLO_MulOp
        (StableHLO_MulOp
          (StableHLO_ExpOp:$e2
            (StableHLO_MulOp
              $x,
              (StableHLO_ConstantLike<"0.5"> $x))),
          (StableHLO_CosineOp:$cs
            (StableHLO_ImagOp:$y $z))),
        $e2),
      (StableHLO_MulOp $e, $cs)),
    (StableHLO_SelectOp
      (StableHLO_CompareOp
       $y,
       (StableHLO_ConstantLike<"0.0">:$zero $x),
        StableHLO_ComparisonDirectionValue<"EQ">,
        (STABLEHLO_DEFAULT_COMPARISON_TYPE)),
      $zero,
      (StableHLO_SelectOp
        $eq_e_constant_posinf,
        (StableHLO_MulOp
          (StableHLO_MulOp
            $e2,
            (StableHLO_SineOp:$sn $y)),
          $e2),
        (StableHLO_MulOp $e, $sn))))>;