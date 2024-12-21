
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
| absolute | complex64 | 967745 | 33704 | 552 | - | - | - |
| absolute<sub>2</sub> | complex64 | 962177 | 39264 | 552 | - | 8 | using native absolute |
| absolute | complex128 | 991861 | 9996 | 144 | - | - | - |
| acos | float32 | 961608 | 38291 | 99 | 3 | - | - |
| acos<sub>2</sub> | float32 | 961878 | 38119 | 4 | - | - | using native acos |
| acos | float64 | 992599 | 7396 | 6 | - | - | - |
| acos | complex64 | 810406 | 190999 | 588 | 8 | - | - |
| acos<sub>2</sub> | complex64 | 678274 | 323467 | 260 | - | - | using native acos |
| acos | complex128 | 750619 | 251104 | 278 | - | - | - |
| acosh | float32 | 988269 | 11704 | 28 | - | - | - |
| acosh<sub>2</sub> | float32 | 999248 | 753 | - | - | - | using native acosh |
| acosh | float64 | 946280 | 53715 | 6 | - | - | - |
| acosh | complex64 | 810406 | 190999 | 588 | 8 | - | - |
| acosh<sub>2</sub> | complex64 | 678274 | 323467 | 260 | - | - | using native acosh |
| acosh | complex128 | 750619 | 251104 | 278 | - | - | - |
| asin | float32 | 974679 | 24368 | 942 | 12 | - | - |
| asin<sub>2</sub> | float32 | 996779 | 3010 | 212 | - | - | using native asin |
| asin | float64 | 995203 | 4772 | 26 | - | - | - |
| asin | complex64 | 808231 | 192742 | 996 | 32 | - | - |
| asin<sub>2</sub> | complex64 | 806919 | 194826 | 252 | 4 | - | using native asin |
| asin | complex128 | 750491 | 250830 | 680 | - | - | - |
| asinh | float32 | 916129 | 83790 | 82 | - | - | - |
| asinh<sub>2</sub> | float32 | 999111 | 890 | - | - | - | using native asinh |
| asinh | float64 | 825119 | 174798 | 84 | - | - | - |
| asinh | complex64 | 808231 | 192742 | 996 | 32 | - | - |
| asinh<sub>2</sub> | complex64 | 806919 | 194826 | 252 | 4 | - | using native asinh |
| asinh | complex128 | 750491 | 250830 | 680 | - | - | - |
| atan | complex64 | 902789 | 97576 | 1636 | - | - | - |
| atan<sub>2</sub> | complex64 | 902789 | 97576 | 1636 | - | - | using native atan |
| atan | complex128 | 960561 | 41364 | 76 | - | - | - |
| atanh | complex64 | 851639 | 145498 | 4780 | 84 | - | - |
| atanh<sub>2</sub> | complex64 | 902789 | 97576 | 1636 | - | - | using native atanh |
| atanh | complex128 | 936441 | 65256 | 304 | - | - | - |
| square | float32 | 997347 | 2654 | - | - | - | - |
| square<sub>2</sub> | float32 | 997347 | 2654 | - | - | - | using native square |
| square | float64 | 999553 | 448 | - | - | - | - |
| square | complex64 | 976865 | 25136 | - | - | - | - |
| square<sub>2</sub> | complex64 | 940185 | 28836 | 16 | - | 32964 | using native square |
| square | complex128 | 995705 | 6296 | - | - | - | - |
| sqrt | float32 | 1000001 | - | - | - | - | using native sqrt |
| sqrt | complex64 | 639573 | 362328 | 100 | - | - | using native sqrt |
| sqrt<sub>2</sub> | complex64 | 644997 | 356784 | 212 | 8 | - | - |
| sqrt | complex128 | 637905 | 364008 | 88 | - | - | - |
| angle | complex64 | 940289 | 61332 | 376 | 4 | - | - |
| angle | complex128 | 989787 | 12214 | - | - | - | - |
| log1p | complex64 | 905931 | 94401 | 1661 | 8 | - | - |
| log1p<sub>2</sub> | complex64 | 899925 | 100395 | 1673 | 8 | - | using fast2sum |
| log1p<sub>3</sub> | complex64 | 698489 | 61706 | 2437 | 1519 | 237850 | using native log1p |
| log1p | complex128 | 846741 | 155252 | 8 | - | - | - |
| tan | float32 | 866723 | 132062 | 1168 | 48 | - | using native tan |
| tan<sub>2</sub> | float32 | 1000001 | - | - | - | - | using upcast tan, native tan |
| real_naive_tan | float32 | 819251 | 179168 | 1580 | 2 | - | - |
| real_naive_tan<sub>2</sub> | float32 | 825895 | 173622 | 484 | - | - | using upcast sin, upcast cos |
| real_naive_tan<sub>3</sub> | float32 | 1000001 | - | - | - | - | using upcast sin, upcast divide, upcast cos |
| tan | complex64 | 783819 | 197524 | 19806 | 792 | 60 | using native tan |
| tan<sub>2</sub> | complex64 | 1001409 | 592 | - | - | - | using upcast tan, native tan |
| tanh | float32 | 985109 | 14892 | - | - | - | using native tanh |
| tanh<sub>2</sub> | float32 | 1000001 | - | - | - | - | using upcast tanh, native tanh |
| tanh | complex64 | 783819 | 197524 | 19806 | 792 | 60 | using native tanh |
| tanh<sub>2</sub> | complex64 | 1001409 | 592 | - | - | - | using upcast tanh, native tanh |
