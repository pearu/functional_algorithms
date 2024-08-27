
# Accuracy of provided algorithms

<sub>This file is generated using estimate_accuracy.py. Do not edit!</sub>

The reference values are obtained by evaluating MPMath functions using
multi-precision arithmetic.

The following table shows the counts of samples that produce function
values being
- different from expected values by the given ULP difference (dULP):
  ```
  ulp_diff(func(sample), reference(sample)) == dULP
  ```

- out of the reference values ulp-range:
  ```
  not (lower <= func(sample) <= upper)
  lower = minimal(reference(s) for s in surrounding(sample) if diff_ulp(s, sample) <= ulp_width)
  upper = maximal(reference(s) for s in surrounding(sample) if diff_ulp(s, sample) <= ulp_width)
  ```

When a counts value is attributed with a superscript, this indicates
the number of samples that lead to out-of-ulp-range results. When dULP
<= 3, out-of-ulp-range counts are acceptable as it typically indicates
that reference function is not sensitive to input perturbations, that
is, `lower == reference(sample) == upper` holds. On the other hand,
when the out-of-ulp-range counts is zero, dULP > 3 counts are
acceptable as it indicates that function's variability is very high
with respect to minimal variations in its input.

When `ulp_width` is specified, its value is indicated as a superscript
in function name. Notice the specified `ulp_width` is not the upper
limit in general: there may exist function-function dependent regions
in complex plane where `ulp_width` needs to be larger to pass the
"out-of-ulp-range counts is zero" test.


| Function | dtype | dULP=0 (exact) | dULP=1 | dULP=2 | dULP=3 | dULP>3 |
| -------- | ----- | -------------- | ------ | ------ | ------ | ------ |
| absolute | complex64 | 967753 | 33696 | 552 | - | - |
| absolute | complex128 | 991753 | 10104 | 144 | - | - |
| acos | float32 | 961608 | 38291 | 99 | 3 | - |
| acos | float64 | 992582 | 7416 | 3 | - | - |
| acos | complex64 | 810108 | 191263 | 622 | 8 | - |
| acos | complex128 | 690209 | 311554 | 238 | - | - |
| acosh | float32 | 988269 | 11704 | 28 | - | - |
| acosh | float64 | 946246 | 53752 | 3 | - | - |
| acosh | complex64 | 810108 | 191263 | 622 | 8 | - |
| acosh | complex128 | 690209 | 311554 | 238 | - | - |
| asin | float32 | 974679 | 24368 | 942 | 12 | - |
| asin | float64 | 995197 | 4776 | 28 | - | - |
| asin | complex64 | 807415 | 193174 | 1320 | 92 | - |
| asin | complex128 | 687179 | 313978 | 844 | - | - |
| asinh | float32 | 916129 | 83790 | 82 | - | - |
| asinh | float64 | 825453 | 174482 | 66 | - | - |
| asinh | complex64 | 807415 | 193174 | 1320 | 92 | - |
| asinh | complex128 | 687179 | 313978 | 844 | - | - |
| atan | float32 | 957469 | 42532 | - | - | - |
| atan | float64 | 992077 | 7924 | - | - | - |
| atan | complex64 | 850741 | 146392 | 4784 | 84 | - |
| atan | complex128 | 936337 | 65264 | 400 | - | - |
| atanh | float32 | 986261 | 13740 | - | - | - |
| atanh | float64 | 995895 | 4106 | - | - | - |
| atanh | complex64 | 850741 | 146392 | 4784 | 84 | - |
| atanh | complex128 | 936337 | 65264 | 400 | - | - |
| square | float32 | 997347 | 2654 | - | - | - |
| square | float64 | 999593 | 408 | - | - | - |
| square | complex64 | 976809 | 25192 | - | - | - |
| square | complex128 | 995833 | 6168 | - | - | - |
| sqrt | float32 | 1000001 | - | - | - | - |
| sqrt | float64 | 1000001 | - | - | - | - |
| sqrt | complex64 | 639749 | 362152 | 100 | - | - |
| sqrt | complex128 | 653493 | 348492 | 16 | - | - |
| angle | complex64 | 940281 | 61338 | 378 | 4 | - |
| angle | complex128 | 989725 | 12276 | - | - | - |
| log1p | float32 | 987438 | 12549 | 14 | - | - |
| log1p | float64 | 945368 | 54622 | 11 | - | - |
| log1p<sup>1</sup> | complex64 | 902287 | 97840<sup>47722</sup> | 1582<sup>1168</sup> | 102<sup>28</sup> | 190<sup>22</sup>!! |
| log1p<sup>2</sup> | complex64 | 902287 | 97840<sup>42698</sup> | 1582<sup>72</sup> | 102<sup>6</sup> | 190<sup>2</sup>!! |
| log1p<sup>3</sup> | complex64 | 902287 | 97840<sup>41454</sup> | 1582<sup>44</sup> | 102 | 190 |
| log1p<sup>1</sup> | complex128 | 801864 | 200067<sup>188447</sup> | 64<sup>10</sup> | 6 | - |
