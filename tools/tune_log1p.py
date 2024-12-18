"""Tools for tuning log1p algorithm.
"""

import functional_algorithms as fa
import numpy as np


def get_reference_func(name, dtype):
    params = fa.utils.function_validation_parameters(name, dtype)
    extra_prec_multiplier = params["extra_prec_multiplier"]
    return getattr(fa.utils.numpy_with_mpmath(extra_prec_multiplier=extra_prec_multiplier, flush_subnormals=False), name)


def main_real():
    """The real part of complex log1p(z)."""

    if 1:
        dtype = np.complex64
        fdtype = np.float32
    else:
        dtype = np.complex128
        fdtype = np.float64

    fi = np.finfo(fdtype)

    params = fa.utils.function_validation_parameters("log1p", dtype)
    max_valid_ulp_count = params["max_valid_ulp_count"]
    max_bound_ulp_width = params["max_bound_ulp_width"]
    extra_prec_multiplier = params["extra_prec_multiplier"]

    log1p = get_reference_func("log1p", dtype)
    log = get_reference_func("log", dtype)

    def log1p1(z):
        """Naive approach, using log.

              Legend:
              (=) - exact match,
              (1) - match with 1 ULP difference,
              (2) - match with 2 ULPs difference,
              (c) - match with less that 100 ULPs difference,
              ( ) - (no)match with more than ULPs difference.

        +inf    =======================================================================================================
        max                                                                                                           =
        1e37                                                                                                          =
        3e35                                                                                                          =
        8e33                                                                                                          =
        2e32                                                                                                          =
        5e30                                                                                                          =
        2e29                                                                                                          =
        4e27                                                                                                          =
        1e26                                                                                                          =
        3e24                                                                                                          =
        9e22                                                                                                          =
        2e21                                                                                                          =
        7e19                                                                                                          =
        2e18    ============================================================================                          =
        5e16    =====================================================================1======                          =
        1e15    ============================================================================                          =
        4e13    ============================================================================                          =
        1e12    ============================================================================                          =
        3e10    ==================================================================1=========                          =
        8e8     ============================================================================                          =
        2e7     ============================================================1===============                          =
        6e5     ============================================================================                          =
        2e4     ============================================================================                          =
        5e2     ============================================================================                          =
        1e1     ============================================1=====1=========================                          =
        4e-1    1111111111111111111111111111111111111111cccccccccc111=======================                          =
        1e-2                                                    cc=1========================                          =
        3e-4                                                   ccc1=========================                          =
        7e-6                                                    cc==========================                          =
        2e-7                                                    cc==========================                          =
        6e-9                                                    cc==========================                          =
        2e-10                                                   cc==========================                          =
        4e-12                                                   cc==========================                          =
        1e-13                                                   cc==========================                          =
        3e-15                                                   cc==========================                          =
        9e-17                                                   cc==========================                          =
        2e-18                                                   cc==========================                          =
        7e-20   =                                               cc==========================                          =
        2e-21   =                                               cc==========================                          =
        5e-23   =                                               cc==========================                          =
        1e-24   =                                               cc==========================                          =
        4e-26   =                                               cc==========================                          =
        1e-27   =                                               cc==========================                          =
        3e-29   =                                               cc==========================                          =
        8e-31   =                                               cc==========================                          =
        2e-32   =                                               cc==========================                          =
        6e-34   =                                               cc==========================                          =
        2e-35   =                                               cc==========================                          =
        4e-37   =                                               cc==========================                          =
        tiny    =                                               cc==========================                          =
        0       =                                               cc==========================                          =
                0     8e-35 3e-30 1e-25 5e-21 2e-16 7e-12 3e-7  1e-2  4e2   2e7   6e11  2e16  9e20  4e25  1e30  5e34  +inf

        """
        x, y = z.real, z.imag
        fptype = x.dtype.type if isinstance(x, np.ndarray) else type(x)

        one = fptype(1)
        half = fptype(0.5)
        return log((x + one) * (x + one) + y * y) * half

    def log1p2(z):
        """Naive approach, using log1p.

        +inf    =======================================================================================================
        max                                                                                                           =
        1e37                                                                                                          =
        3e35                                                                                                          =
        8e33                                                                                                          =
        2e32                                                                                                          =
        5e30                                                                                                          =
        2e29                                                                                                          =
        4e27                                                                                                          =
        1e26                                                                                                          =
        3e24                                                                                                          =
        9e22                                                                                                          =
        2e21                                                                                                          =
        7e19                                                                                                          =
        2e18    ============================================================================                          =
        5e16    =====================================================================1======                          =
        1e15    ============================================================================                          =
        4e13    ============================================================================                          =
        1e12    ============================================================================                          =
        3e10    ==================================================================1=========                          =
        8e8     ============================================================================                          =
        2e7     ============================================================1===============                          =
        6e5     ============================================================================                          =
        2e4     ============================================================================                          =
        5e2     ============================================================================                          =
        1e1     ============================================1=====1==1======================                          =
        4e-1    =========================================1==1=11=1==1=======================                          =
        1e-2    =====================================1=1=1===1111=1=========================                          =
        3e-4    ====================================1=1==1==1111=1=1========================                          =
        7e-6    ===============================1=====1====1=1===1=11========================                          =
        2e-7    =======================1===1====1==========111==1=11========================                          =
        6e-9    ===================1=1=1=1===============11111==1=11========================                          =
        2e-10   ===============1===1=====1=1=============11111==1=11========================                          =
        4e-12   ===========1111111==1====================11111==1=11========================                          =
        1e-13   =========================================11111==1=11========================                          =
        3e-15   ======11=1===============================11111==1=11========================                          =
        9e-17   ==1===11=1===============================11111==1=11========================                          =
        2e-18   ==111====================================11111==1=11========================                          =
        7e-20   =========================================11111==1=11========================                          =
        2e-21   ==1======================================11111==1=11========================                          =
        5e-23   =========================================11111==1=11========================                          =
        1e-24   =========================================11111==1=11========================                          =
        4e-26   =========================================11111==1=11========================                          =
        1e-27   =========================================11111==1=11========================                          =
        3e-29   =========================================11111==1=11========================                          =
        8e-31   =========================================11111==1=11========================                          =
        2e-32   =========================================11111==1=11========================                          =
        6e-34   =========================================11111==1=11========================                          =
        2e-35   =========================================11111==1=11========================                          =
        4e-37   =========================================11111==1=11========================                          =
        tiny    =========================================11111==1=11========================                          =
        0       =========================================11111==1=11========================                          =
                0     8e-35 3e-30 1e-25 5e-21 2e-16 7e-12 3e-7  1e-2  4e2   2e7   6e11  2e16  9e20  4e25  1e30  5e34  +inf
        """
        x, y = z.real, z.imag
        fptype = x.dtype.type if isinstance(x, np.ndarray) else type(x)

        half = fptype(0.5)
        return log1p(x + x + y * y + x * x) * half

    def log1p3(z):
        """Using log1p with Dekker's product.

        +inf
        max
        1e37
        3e35
        8e33
        2e32
        5e30
        2e29
        4e27
        1e26
        3e24
        9e22
        2e21
        7e19
        2e18    ============================================================================
        5e16    =====================================================================1======
        1e15    ============================================================================
        4e13    ============================================================================
        1e12    ============================================================================
        3e10    ==================================================================1=========
        8e8     ============================================================================
        2e7     ============================================================1===============
        6e5     ============================================================================
        2e4     ============================================================================
        5e2     ============================================================================
        1e1     ============================================1=====1==1======================
        4e-1    =========================================1==1=11=1==1=======================
        1e-2    =====================================1=1=1===1111=1=========================
        3e-4    ====================================1=1==1==1111=1=1========================
        7e-6    ===============================1=====1====1=1===1=11========================
        2e-7    =======================1===1====1==========111==1=11========================
        6e-9    ===================1=1=1=1===============11111==1=11========================
        2e-10   ===============1===1=====1=1=============11111==1=11========================
        4e-12   ===========1111111==1====================11111==1=11========================
        1e-13   =========================================11111==1=11========================
        3e-15   ======11=1===============================11111==1=11========================
        9e-17   ==1===11=1===============================11111==1=11========================
        2e-18   11=11====================================11111==1=11========================
        7e-20   =========================================11111==1=11========================
        2e-21   ==1======================================11111==1=11========================
        5e-23   =========================================11111==1=11========================
        1e-24   =========================================11111==1=11========================
        4e-26   =========================================11111==1=11========================
        1e-27   =========================================11111==1=11========================
        3e-29   =========================================11111==1=11========================
        8e-31   =========================================11111==1=11========================
        2e-32   =========================================11111==1=11========================
        6e-34   =========================================11111==1=11========================
        2e-35   =========================================11111==1=11========================
        4e-37   =========================================11111==1=11========================
        tiny    =========================================11111==1=11========================
        0       =========================================11111==1=11========================
                0     8e-35 3e-30 1e-25 5e-21 2e-16 7e-12 3e-7  1e-2  4e2   2e7   6e11  2e16  9e20  4e25  1e30  5e34  +inf
        """
        x, y = z.real, z.imag
        fptype = x.dtype.type if isinstance(x, np.ndarray) else type(x)

        half = fptype(0.5)
        one = fptype(1.0)

        C = fa.utils.get_veltkamp_splitter_constant(fptype)
        # xh, xl = fa.utils.split_veltkamp(x, C=C)
        xh, xl = fa.utils.add_2sum(x, x)
        xxh, xxl = fa.utils.square_dekker(x, C=C)
        yyh, yyl = fa.utils.square_dekker(y, C=C)
        a = fa.utils.sum_2sum([xh, yyh, xxh, yyl, xxl])[0]
        return log1p(a) * half

    def log1p4(z):
        """Large inputs.
                log((1+x)**2 + y**2) ~= log(x**2 + y**2) = log(mx**2 * (1 + mn**2/mx**2)) = 2*log(mx) + log1p((mn/mx)**2)

        +inf    ======================================================================================================
        max     =======================================================================================================
        1e37    =================================================================================================1=1===
        3e35    ===============================================================================================1==11===
        8e33    ============================================================================================1==1=======
        2e32    ==============================================================================================1========
        5e30    =======================================================================================================
        2e29    =========================================================================================1==1==========
        4e27    ========================================================================================11=============
        1e26    =================================================================================11==11================
        3e24    ===================================================================================1===================
        9e22    ==============================================================================111======================
        2e21    =======================================================================================================
        7e19    ================================================================================1======================
        2e18    =======================================================================1===============================
        5e16    ========================================================================1===1==========================
        1e15    =======================================================================================================
        4e13    ==================================================================1====================================
        1e12    =================================================================1=====================================
        3e10    =================================================================1=====================================
        8e8     =============================================================1=11=1====================================
        2e7     =========================================================11=111========================================
        6e5     =========================================================111=11========================================
        2e4     ======================================================2ccc2============================================
        5e2     cccccccccccccccccccccccccccccccccccccccccccccccccccc    ccc1===========================================
        1e1                                                             cc21===========================================
        4e-1                                                            cc21===========================================
        1e-2                                                            cc21===========================================
        3e-4                                                            cc21===========================================
        7e-6                                                            cc21===========================================
        2e-7                                                            cc21===========================================
        6e-9                                                            cc21===========================================
        2e-10                                                           cc21===========================================
        4e-12                                                           cc21===========================================
        1e-13                                                           cc21===========================================
        3e-15                                                           cc21===========================================
        9e-17                                                           cc21===========================================
        2e-18                                                           cc21===========================================
        7e-20                                                           cc21===========================================
        2e-21                                                           cc21===========================================
        5e-23                                                           cc21===========================================
        1e-24                                                           cc21===========================================
        4e-26                                                           cc21===========================================
        1e-27                                                           cc21===========================================
        3e-29                                                           cc21===========================================
        8e-31                                                           cc21===========================================
        2e-32                                                           cc21===========================================
        6e-34                                                           cc21===========================================
        2e-35                                                           cc21===========================================
        4e-37                                                           cc21===========================================
        tiny                                                            cc21===========================================
        0                                                               cc21===========================================
                0     8e-35 3e-30 1e-25 5e-21 2e-16 7e-12 3e-7  1e-2  4e2   2e7   6e11  2e16  9e20  4e25  1e30  5e34  +inf
        """
        x, y = z.real, z.imag
        fptype = x.dtype.type if isinstance(x, np.ndarray) else type(x)

        mx = np.fmax(abs(x), abs(y))
        mn = np.fmin(abs(x), abs(y))
        r = mn / mx  # if mn == mx == inf, r should be evaluated as 1
        half = fptype(0.5)
        return log(mx) + log1p(r * r) * half

    def log1p5(z):
        """Large and close real and imag parts of z.
              log((1+x)**2 + y**2) ~= log(x**2 + y**2) = log(x**2 + (y - x + x)**2)
              = log(x**2 + (y - x)**2 + 2 * (y-x)*x + x**2)
              = log(2*y*x + (y-x)**2) = log(2*x*y*(1 + ((y-x)**2/(x*y*2))))

        +inf
        max                                                        =========================1====================1====
        1e37                                                     =111=1==11===1=======1====1===11111==111=1===1====1=1
        3e35                                                   ==========================================1=1=1=1===1==
        8e33                                                 ============================================1=1=1====1===
        2e32                                               ===1====================================1===1=======11==11=
        5e30                                             =1==1=1==111=1===1==========11====1=1=11111==111=1=1==11===1=
        2e29                                           =================================================11=1======1===
        4e27                                         =1==1======111==1=11==11==1111111=1===1===1=1======1=1===========
        1e26                                       =======================================1====1=11=1=1===1===1======1
        3e24                                     =====================================1====1===1111==================1
        9e22                                   11=1===1==1==1=====1==1==1==11====11==1=1===1==========================
        2e21                                 =========================================1========1111===================
        7e19                               ===========================================1========1=11===1===============
        2e18                             =============================================1=========1===1=================
        5e16                           1=1==11=11=1===============1==1=====1======1==1==1==========1=========1========
        1e15                        =======================================================1=1=1111===1===============
        4e13                      111111111111111=11==1=11====111==11==11=11=========1================1===========1===
        1e12                    1=1==111=1===11============================1=1=====1===1=====1=====1==================
        3e10                  ===1=11===1=111=1=============================1111==11=====111==11=11=1=1===============
        8e8                 111111111111111111111=11==1=11=======1=11===1=1==1==111==============11===1===========1===
        2e7               =======1=11=======1=====11=====1====================1==1========1==1=1111===1===========1===
        6e5             22222222222222=2211111111=11==1==1===============1=1=11===1===1==111=111111=1=================
        2e4           ==c========2=2222=====2=2==1==111==1============2ccc21===1=1===========1====1=1=1===========1===
        5e2         cccccccccccccc22ccc2c22ccccccccccccccccccccccccc    ccc2======1==========1====================1===
        1e1                                                             cc21======1=======1==1===================11===
        4e-1                                                            cc21==========1==111=111111=1================
        1e-2                                                            cc21======1=======1==1========1===========1
        3e-4                                                            cc2=======1==============================
        7e-6                                                            cc2=======1==========1=================
        2e-7                                                            cc21==========1===1==1===============
        6e-9                                                            cc2=======1================1=======
        2e-10                                                           cc12======1================1=1===
        4e-12                                                           cc21===========================
        1e-13                                                           ccc2======1=======1==11=1===1
        3e-15                                                           cc21=======================
        9e-17                                                           cc21=====================
        2e-18                                                           cc21=================1=
        7e-20                                                           cc21=================
        2e-21                                                           cc2=======1===1==11
        5e-23                                                           cc22==2==========
        1e-24                                                           ccc2===========
        4e-26                                                           ccc2=========
        1e-27                                                           cc=2=======
        3e-29                                                           cc222==2=
        8e-31                                                           cc22==
        2e-32                                                           cc22
        6e-34                                                           cc
        2e-35
        4e-37
        tiny
        0
                0     8e-35 3e-30 1e-25 5e-21 2e-16 7e-12 3e-7  1e-2  4e2   2e7   6e11  2e16  9e20  4e25  1e30  5e34  +inf
        """
        x, y = z.real, z.imag
        fptype = x.dtype.type if isinstance(x, np.ndarray) else type(x)

        half = fptype(0.5)
        r = (y - x) / x / y * half * (y - x)
        return (log(fptype(2.0)) + log(x) + log(y) + log1p(r)) * half

    def log1p6(z):
        x, y = z.real, z.imag
        fptype = x.dtype.type if isinstance(x, np.ndarray) else type(x)

        half = fptype(0.5)
        one = fptype(1.0)

        C = fa.utils.get_veltkamp_splitter_constant(fptype)

        xh, xl = fa.utils.add_2sum(x, x)
        xxh, xxl = fa.utils.square_dekker(x, C=C)
        yyh, yyl = fa.utils.square_dekker(y, C=C)
        a = fa.utils.sum_2sum([xh, yyh, xxh, yyl, xxl])[0]
        return log1p(a) * half

        xh, x2l = fa.utils.add_2sum(x, x)
        xxh, xxl = fa.utils.square_dekker(x, C=C)
        yyh, yyl = fa.utils.square_dekker(y, C=C)
        a = fa.utils.sum_2sum([xh, yyh, xxh, yyl, xxl])[0]
        return log1p(a) * half

    size = 51

    log1p_approx = log1p1

    """
    Conclusion:
      use log1p4 for large x or y,
      use log1p1 for x close to -1 and y is not large
      otherwise use log1p3
    """

    if 0:
        min_real_value = None
        max_real_value = None
        min_imag_value = None
        max_imag_value = None
    else:
        # study the surrounding of the given point:
        z, dz = -4e-38 + 2e-19j, 3.9999999e-38 + 1e-19j
        z, dz = -0.996 + 0.0002j, 0.01 + 0.0001j
        min_real_value = z.real - dz.real
        max_real_value = z.real + dz.real
        min_imag_value = z.imag - dz.imag
        max_imag_value = z.imag + dz.imag

        if 0:
            min_real_value, max_real_value = -1e-37, -1e-39
            min_imag_value, max_imag_value = 1.5e-19, 2.5e-19
        if 0:
            min_real_value, max_real_value = -1.001, -0.999
            min_imag_value, max_imag_value = 0, 0.01
        if 0:
            min_real_value, max_real_value = 0, np.sqrt(fi.max) * 0.05
            min_imag_value, max_imag_value = 0, np.sqrt(fi.max) * 0.99

        if 1:
            min_real_value, max_real_value = -1.9, -0.5
            min_imag_value, max_imag_value = 0, np.sqrt(fi.max)

        if 1:
            min_real_value, max_real_value = -2.1, -1.9
            min_imag_value, max_imag_value = 0, np.sqrt(fi.max)

    samples = fa.utils.complex_samples(
        (2 * size, size * 1),
        dtype=dtype,
        include_huge=False,
        include_subnormal=False,
        nonnegative=True,
        min_real_value=min_real_value,
        max_real_value=max_real_value,
        min_imag_value=min_imag_value,
        max_imag_value=max_imag_value,
    )[::-1]

    ref = log1p.call(samples).real

    approx = log1p_approx(samples)
    diff = fa.utils.diff_ulp(ref, approx)

    timage = fa.TextImage()
    timage.fill(0, 10, diff == 0, symbol="=")
    timage.fill(0, 10, diff == 1, symbol="1")
    timage.fill(0, 10, diff == 2, symbol="2")
    timage.fill(0, 10, (diff > 2) & (diff < 1000000), symbol="c")

    real_axis = samples[0:1].real
    imag_axis = samples[:, 0:1].imag
    timage.insert(0, 2, fa.TextImage.fromseq(imag_axis))
    timage.append(-1, 10, fa.TextImage.fromseq(real_axis[:, ::7], mintextwidth=6, maxtextwidth=6))
    print(timage)

    if 1:
        # mask = (diff > 1000000) & (abs(samples.imag - fdtype(2e-19)) < 1e-22)
        mask = diff > 3
        print(approx[mask][:10])
        print(ref[mask][:10])
        print(samples[mask][:10])

    print(np.finfo(fdtype))


if __name__ == "__main__":
    main_real()