// This file is generated using functional_algorithms tool (0.11.0), see
//   https://github.com/pearu/functional_algorithms
// for more information.




def : Pat<(atan_1 ComplexElementType:$z),
  (StableHLO_ComplexOp
    (StableHLO_ImagOp
      (CHLO_AtanhOp:$w
        (StableHLO_ComplexOp
          (StableHLO_NegOp
            (StableHLO_ImagOp $z)),
          (StableHLO_RealOp $z)))),
    (StableHLO_NegOp
      (StableHLO_RealOp $w)))>;