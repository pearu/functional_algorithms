
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

Finally,
- "using native <function>" means "using the corresponding numpy <function>",
- "using upcast <function>" means that the function arguments are
  upcasted to a dtype with bits doubled, and the function results are
  downcasted to a dtype with bits split half.

| Function | dtype | dULP=0 (exact) | dULP=1 | dULP=2 | dULP=3 | dULP>3 | Notes |
| -------- | ----- | -------------- | ------ | ------ | ------ | ------ | ----- |
| absolute | complex64 | 967753 | 33696 | 552 | - | - | - |
| absolute<sub>2</sub> | complex64 | 962185 | 39256 | 552 | - | 8 | using native absolute |
| absolute | complex128 | 991753 | 10104 | 144 | - | - | - |
| acos | float32 | 961608 | 38291 | 99 | 3 | - | - |
| acos<sub>2</sub> | float32 | 961878 | 38119 | 4 | - | - | using native acos |
| acos | float64 | 992582 | 7416 | 3 | - | - | - |
| acos | complex64 | 810108 | 191263 | 622 | 8 | - | - |
| acos<sub>2</sub> | complex64 | 678888 | 322829 | 284 | - | - | using native acos |
| acos | complex128 | 690209 | 311554 | 238 | - | - | - |
| acosh | float32 | 988269 | 11704 | 28 | - | - | - |
| acosh<sub>2</sub> | float32 | 999248 | 753 | - | - | - | using native acosh |
| acosh | float64 | 946246 | 53752 | 3 | - | - | - |
| acosh | complex64 | 810108 | 191263 | 622 | 8 | - | - |
| acosh<sub>2</sub> | complex64 | 678888 | 322829 | 284 | - | - | using native acosh |
| acosh | complex128 | 690209 | 311554 | 238 | - | - | - |
| asin | float32 | 974679 | 24368 | 942 | 12 | - | - |
| asin<sub>2</sub> | float32 | 996779 | 3010 | 212 | - | - | using native asin |
| asin | float64 | 995197 | 4776 | 28 | - | - | - |
| asin | complex64 | 807415 | 193174 | 1320 | 92 | - | - |
| asin<sub>2</sub> | complex64 | 807661 | 194084 | 252 | 4 | - | using native asin |
| asin | complex128 | 687179 | 313978 | 844 | - | - | - |
| asinh | float32 | 916129 | 83790 | 82 | - | - | - |
| asinh<sub>2</sub> | float32 | 999111 | 890 | - | - | - | using native asinh |
| asinh | float64 | 825453 | 174482 | 66 | - | - | - |
| asinh | complex64 | 807415 | 193174 | 1320 | 92 | - | - |
| asinh<sub>2</sub> | complex64 | 807661 | 194084 | 252 | 4 | - | using native asinh |
| asinh | complex128 | 687179 | 313978 | 844 | - | - | - |
| atan | complex64 | 850741 | 146392 | 4784 | 84 | - | - |
| atan<sub>2</sub> | complex64 | 903333 | 97028 | 1640 | - | - | using native atan |
| atan | complex128 | 936337 | 65264 | 400 | - | - | - |
| atanh | complex64 | 850741 | 146392 | 4784 | 84 | - | - |
| atanh<sub>2</sub> | complex64 | 903333 | 97028 | 1640 | - | - | using native atanh |
| atanh | complex128 | 936337 | 65264 | 400 | - | - | - |
| square | float32 | 997347 | 2654 | - | - | - | - |
| square<sub>2</sub> | float32 | 997347 | 2654 | - | - | - | using native square |
| square | float64 | 999593 | 408 | - | - | - | - |
| square | complex64 | 976809 | 25192 | - | - | - | - |
| square<sub>2</sub> | complex64 | 939997 | 29016 | 16 | - | 32972 | using native square |
| square | complex128 | 995833 | 6168 | - | - | - | - |
| sqrt | float32 | 1000001 | - | - | - | - | using native sqrt |
| sqrt | complex64 | 639749 | 362152 | 100 | - | - | using native sqrt |
| angle | complex64 | 940281 | 61338 | 378 | 4 | - | - |
| angle | complex128 | 989725 | 12276 | - | - | - | - |
| log1p<sup>1</sup> | complex64 | 902287 | 97840<sup>47722</sup> | 1582<sup>1168</sup> | 102<sup>28</sup> | 190<sup>22</sup>!! | - |
| log1p<sup>2</sup> | complex64 | 902287 | 97840<sup>42698</sup> | 1582<sup>72</sup> | 102<sup>6</sup> | 190<sup>2</sup>!! | - |
| log1p<sup>3</sup> | complex64 | 902287 | 97840<sup>41454</sup> | 1582<sup>44</sup> | 102 | 190 | - |
| log1p<sup>1</sup> | complex64 | 697224 | 62971<sup>50269</sup> | 2437<sup>1108</sup> | 1521<sup>1505</sup> | 237848<sup>237612</sup>!! | using native log1p |
| log1p<sup>2</sup>[using={'native log1p'}] | complex64 | 697224 | 62971<sup>49386</sup> | 2437<sup>48</sup> | 1521<sup>984</sup> | 237848<sup>237376</sup>!! | using native log1p |
| log1p<sup>3</sup>[using={'native log1p'}] | complex64 | 697224 | 62971<sup>49206</sup> | 2437<sup>8</sup> | 1521<sup>36</sup> | 237848<sup>235730</sup>!! | using native log1p |
| log1p<sup>1</sup> | complex128 | 801864 | 200067<sup>188447</sup> | 64<sup>10</sup> | 6 | - | - |
| tan | float32 | 866723 | 132062 | 1168 | 48 | - | using native tan |
| tan<sub>2</sub> | float32 | 1000001 | - | - | - | - | using native tan, upcast tan |
| tan | complex64 | 783679 | 197584 | 19902 | 776 | 60 | using native tan |
| tan<sub>2</sub> | complex64 | 1001417 | 584 | - | - | - | using native tan, upcast tan |
| tanh | float32 | 985109 | 14892 | - | - | - | using native tanh |
| tanh<sub>2</sub> | float32 | 1000001 | - | - | - | - | using native tanh, upcast tanh |
| tanh | complex64 | 783679 | 197584 | 19902 | 776 | 60 | using native tanh |
| tanh<sub>2</sub> | complex64 | 1001417 | 584 | - | - | - | using native tanh, upcast tanh |
| real_naive_tan | float32 | 819251 | 179168 | 1580 | 2 | - | - |
| real_naive_tan<sub>2</sub> | float32 | 825895 | 173622 | 484 | - | - | using upcast cos, upcast sin |
