
# Accuracy of provided algorithms

The following table shows the counts of samples that produce function
values being different from expected values by the given ULP
difference (dULP). The expected values are obtained by evaluating
MPMath functions using multi-precision arithmetic.

| Function | dtype | dULP=0 (exact) | dULP=1 | dULP=2 | dULP=3 | dULP>3 | errors    |
| -------- | ----- | ------------- | ----- | ----- | ----- | ----- | --------- |
| absolute | float32 | 1000001 | - | - | - | - | - |
| absolute | float64 | 1000001 | - | - | - | - | - |
| absolute | complex64 | 977109 | 24892 | - | - | - | - |
| absolute | complex128 | 989613 | 12372 | 16 | - | - | - |
| acos | float32 | 504842 | 488335 | 6599 | 225 | - | - |
| acos | float64 | 985299 | 13318 | 1376 | 8 | - | - |
| acos | complex64 | 804838 | 196477 | 680 | 6 | - | - |
| acos | complex128 | 701506 | 300255 | 238 | 2 | - | - |
| asin | float32 | 937353 | 61812 | 810 | 26 | - | - |
| asin | float64 | 983317 | 16662 | 22 | - | - | - |
| asin | complex64 | 808487 | 191878 | 1592 | 44 | - | - |
| asin | complex128 | 694103 | 307326 | 560 | 12 | - | - |
| asinh | float32 | 922791 | 77144 | 66 | - | - | - |
| asinh | float64 | 829637 | 170270 | 94 | - | - | - |
| asinh | complex64 | 767789 | 233968 | 244 | - | - | - |
| asinh | complex128 | 850261 | 151496 | 244 | - | - | - |
| square | float32 | 997293 | 2708 | - | - | - | - |
| square | float64 | 999649 | 352 | - | - | - | - |
| square | complex64 | 974577 | 27424 | - | - | - | - |
| square | complex128 | 994505 | 7496 | - | - | - | - |
