
# Accuracy of provided algorithms

The following table shows the counts of samples that produce function
values being different from expected values by the given ULP
difference (dULP). The expected values are obtained by evaluating
MPMath functions using multi-precision arithmetic.

| Function | dtype | dULP=0 (exact) | dULP=1 | dULP=2 | dULP=3 | dULP>3 | errors    |
| -------- | ----- | ------------- | ----- | ----- | ----- | ----- | --------- |
| absolute | float32 | 1000001 | - | - | - | - | - |
| absolute | float64 | 1000001 | - | - | - | - | - |
| absolute | complex64 | 967753 | 33696 | 552 | - | - | - |
| absolute | complex128 | 991753 | 10104 | 144 | - | - | - |
| acos | float32 | 548396 | 444291 | 7072 | 242 | - | - |
| acos | float64 | 985930 | 12727 | 1338 | 6 | - | - |
| acos | complex64 | 810108 | 191263 | 622 | 8 | - | - |
| acos | complex128 | 690209 | 311554 | 238 | - | - | - |
| acosh | float32 | 988269 | 11704 | 28 | - | - | - |
| acosh | float64 | 946246 | 53752 | 3 | - | - | - |
| acosh | complex64 | 810108 | 191263 | 622 | 8 | - | - |
| acosh | complex128 | 690209 | 311554 | 238 | - | - | - |
| asin | float32 | 974679 | 24368 | 942 | 12 | - | - |
| asin | float64 | 995197 | 4776 | 28 | - | - | - |
| asin | complex64 | 807415 | 193174 | 1320 | 92 | - | - |
| asin | complex128 | 687179 | 313978 | 844 | - | - | - |
| asinh | float32 | 916129 | 83790 | 82 | - | - | - |
| asinh | float64 | 825453 | 174482 | 66 | - | - | - |
| asinh | complex64 | 807415 | 193174 | 1320 | 92 | - | - |
| asinh | complex128 | 687179 | 313978 | 844 | - | - | - |
| square | float32 | 997347 | 2654 | - | - | - | - |
| square | float64 | 999593 | 408 | - | - | - | - |
| square | complex64 | 976809 | 25192 | - | - | - | - |
| square | complex128 | 995833 | 6168 | - | - | - | - |
