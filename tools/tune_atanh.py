"""Tools for tuning atanh algorithm.
"""

import functional_algorithms as fa
import numpy as np


def unarray(func):

    def wrapper(*args):
        return func(*args)[()]

    return wrapper


def get_reference_func(name, dtype):
    params = fa.utils.function_validation_parameters(name, dtype)
    extra_prec_multiplier = params["extra_prec_multiplier"]
    return getattr(fa.utils.numpy_with_mpmath(extra_prec_multiplier=extra_prec_multiplier), name)


def bisect(pred, a, b, dtype=None, extra_args=()):
    """find x such that

      pred(prev(x)) == pred(x) and pred(x) != pred(next(x))

    or

     pred(prev(x)) != pred(x) and pred(x) == pred(next(x))

    within interval [a, b].

    pred(a) != pred(b) must hold
    """
    if dtype is None:
        dtype = type(a)
    if a > b:
        a, b = b, a
    pa = pred(a, *extra_args)
    pb = pred(b, *extra_args)
    assert pa != pb, (a, b, pa, pb)
    while True:
        m = dtype((a + b) / 2)
        if m == a or m == b:
            return m
        pm = pred(m, *extra_args)
        if pa != pm:
            b = m
            pb = pm
        elif pb != pm:
            a = m
            pa = pm
        else:
            assert 0, (a, b, m)


def maximal_bisect(pred, a, b, mn, mx, dtype):
    """find x such that

       bisect(pred, mn, mx, extra_args=(x,))

    result has a maximal possible value whereas a < x and x < b.

    The maximal value must exist in range [mn, mx].
    """
    a, b = dtype(a), dtype(b)
    fa = bisect(pred, mn, mx, dtype, extra_args=(a,))
    fb = bisect(pred, mn, mx, dtype, extra_args=(b,))

    while True:
        m1 = a + (b - a) / 3
        m2 = dtype((m1 + b) / 2)
        m1 = dtype(m1)
        f1 = bisect(pred, mn, mx, dtype, extra_args=(m1,))
        if m1 == m2:
            return f1
        f2 = bisect(pred, mn, mx, dtype, extra_args=(m2,))
        if f1 > f2:
            a, b = a, m2
            fa, fb = fa, f2
        else:
            a, b = m1, b
            fa, fb = f1, fb
        mn = min(fa, fb)


def main_imag():
    """The imaginary part of complex atanh(z).

    Observe the mask of fp(abs(imag(atanh(z)))) == fp(pi/2) on the
    first quarted of complex plane (0<=z.real<inf, 0<=z.imag<inf):

    ^  log(imag)
    |
    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    xxxxxxxxxxxxxxxxxxxZxxxxxxxxxxxx
                       xxxxxxxxxxxxx
                      xxxxxxxxxxxxxx
                      xxxxxxxxxxxxxx
                     xxxxxxxxxxxxxxx
                     xxxxxxxxxxxxxxx
                    xxxxxxxxxxxxxxxx
                    xxxxxxxxxxxxxxxx
                    xxxxxxxxxxxxxxxx
                    xxxxxxxxxxxxxxxx
                    xxxxxxxxxxxxxxxx
                    xxxxxxxxxxxxxxxx
                    xxxxxxxxxxxxxxxx
                    xxxxxxxxxxxxxxxx
                    xxxxxxxxxxxxxxxx
                    xxxxxxxxxxxxxxxx
                    xxxxxxxxxxxxxxxx
                    xxxxxxxxxxxxxxxx
                    xxxxxxxxxxxxxxxx
    o               xxxxxxxxxxxxxxxx  -> log(real)

    This shows that there exists parameters X and Y (or Z=X+I*Y as
    marked in the mask representation above) such that

      atanh(z) == pi / 2

    for any z such that z.imag > Y or if z.real > X where the equality
    `==` is defined as the equality of floating point numbers.

    In the following we'll find the parameters X and Y for different
    floating point systems:

      float16: X, Y=(514.0, 1028.0)
      float32: X, Y=(31459888.0, 62919776.0)
      float64: X, Y=(2902679387770655.0, 5805358775541310.0)
      longdouble: X, Y=(1.716601425640972592e+19, 3.433202851281945184e+19)

    Notice the following relations:

      Y is approximately 2 * X

      1 / eps < Y < 8 / eps

    for all fp types where

      eps = nextafter(fp(1), fp(2)) - fp(1)
    """

    dtype = np.complex64
    fdtype = np.float32

    params = fa.utils.function_validation_parameters("atanh", dtype)
    max_valid_ulp_count = params["max_valid_ulp_count"]
    max_bound_ulp_width = params["max_bound_ulp_width"]
    extra_prec_multiplier = params["extra_prec_multiplier"]
    func = getattr(
        fa.utils.numpy_with_mpmath(extra_prec_multiplier=extra_prec_multiplier, flush_subnormals=True),
        "atanh",
    )

    size = 31
    samples = fa.utils.complex_samples(
        (size, size), dtype=dtype, include_huge=False, include_subnormal=False, nonnegative=True
    )[::-1]

    values = func.call(samples)

    mask = np.float32(abs(values.imag)) == np.float32(np.pi / 2)

    timage = fa.TextImage()
    timage.fill(0, 0, mask, symbol="x")
    print(timage)

    # compute X and Y for
    for fdtype in [np.float16, np.float32, np.float64, np.float128]:
        dtype = {np.float16: np.complex64, np.float32: np.complex64, np.float64: np.complex128, np.float128: np.complex256}[
            fdtype
        ]

        largest = np.finfo(fdtype).max / 8
        target = fdtype("1.5707963267948966192313216916397514420985846996875529104874722")  # pi / 2

        def f2(x, y):
            z = np.empty((), dtype=dtype)
            z.real = x
            z.imag = y
            r = func(z).imag
            imag = fdtype(r)
            return imag == target

        def f1(y):
            return f2(0, y)

        Y = bisect(f1, fdtype(100), largest, fdtype)
        X = maximal_bisect(f2, 0.1, Y * 0.9, 1, Y, fdtype)

        print(f"{fdtype.__name__}: {X, Y=}", "8 / eps = %e" % (8 / np.finfo(fdtype).eps))


def functional_nextafter(func, x, d):
    """Return x1 such that

    func(nextafter(x1, x)) == func(x)
    func(x1) != func(x)
    x1 > x if right else x1 < x
    """
    dtype = type(x)
    f = func(x)
    x1 = x
    oo = dtype(d)
    while func(x1) == f:
        x1 = np.nextafter(x1, oo)
    assert func(np.nextafter(x1, x)) == f
    assert func(x1) != f
    return x1


def main_real():
    """The real part of complex atanh(z)."""

    dtype = np.complex128
    fdtype = np.float64

    params = fa.utils.function_validation_parameters("atanh", dtype)
    max_valid_ulp_count = params["max_valid_ulp_count"]
    max_bound_ulp_width = params["max_bound_ulp_width"]
    extra_prec_multiplier = params["extra_prec_multiplier"]

    atanh = get_reference_func("atanh", dtype)
    log1p = get_reference_func("log1p", dtype)
    log = get_reference_func("log", dtype)

    def atanh1(z):
        """Naive approach. The real part of the single precision float atanh
              on the first quarter of complex plane compared to reference
              atanh(z).real. Legend: (=) - exact match, (1) - match with 1
              ULP difference, (2) - match with 2 ULPs difference, (c) -
              match with less that 100 ULPs difference, ( ) - (no)match with
              more than ULPs difference.

        +inf
        max
        8e35
        2e33
        4e30
        1e28
        2e25
        5e22
        1e20
        3e17    =============================
        7e14    =========================
        2e12    =====================
        4e9     ================
        8e6     ============
        2e4     ========
        4e1     ====                          cccc
        1e-1    ==                           c2=cc
        2e-4    ==                           c21cc
        5e-7    =1                           c1=cc
        1e-9    =1                           c1=cc
        3e-12   =1                           c1=cc
        6e-15   =1                           c1=cc
        1e-17   =1                           c1=cc
        4e-20   =1                           c1=cc
        8e-23   =1                           c1=cc
        2e-25   =1                           c1=cc
        4e-28   =1                           c1=cc
        1e-30   =1                           c1=cc
        2e-33   =1                           c1=cc
        5e-36   =1                           c1=cc
        tiny    =1                           c1=cc
        0       =1                           c1=cc
                0     3e-32 1e-24 5e-17 3e-9  1e-1  5e6   2e14  9e21  4e29  2e37

        """
        x, y = z.real, z.imag
        fptype = x.dtype.type if isinstance(x, np.ndarray) else type(x)

        one = fptype(1)
        quarter = fptype(0.25)
        return log(((x + one) * (x + one) + y * y) / ((x - one) * (x - one) + y * y)) * quarter

    def atanh2(z):
        """
        +inf    ==============================================================
        max     ==============================================================
        8e35    ==========================================================   =
        2e33    ======================================================       =
        4e30    ==================================================           =
        1e28    =============================================                =
        2e25    =========================================                    =
        5e22    =====================================                        =
        1e20    =================================                            =
        3e17    =====================================1=======1               =
        7e14    ===========================1========1======11=               =
        2e12    ====================1==1====1====1==========1=               =
        4e9     ===================1===11=1=1====111====1=111=               =
        8e6     ============1==111=111=111==111=11==111===111=               =
        2e4     =========1========111===11=1=1=====11=====111=               =
        4e1     ========1==1=1==1=11==1=1==11=======1=====111=               =
        1e-1    =11111===1==1111111=111===1======1==1=====111=               =
        2e-4    =111111111=111=11111111=111==1===1==1=====111=               =
        5e-7    =========================11=111==1==1=====111=               =
        1e-9    =========================11=111==1==1=====111=               =
        3e-12   =========================11=111==1==1=====111=               =
        6e-15   =========================11=111==1==1=====111=               =
        1e-17   =========================11=111==1==1=====111=               =
        4e-20   =========================11=111==1==1=====111=               =
        8e-23   =========================11=111==1==1=====111=               =
        2e-25   =========================11=111==1==1=====111=               =
        4e-28   =========================11=111==1==1=====111=               =
        1e-30   =========================11=111==1==1=====111=               =
        2e-33   =========================11=111==1==1=====111=               =
        5e-36   =========================11=111==1==1=====111=               =
        tiny    =========================11=111==1==1=====111=               =
        0       =========================11=111==1==1=====111=               =
                0     3e-32 1e-24 5e-17 3e-9  1e-1  5e6   2e14  9e21  4e29  2e37
        """
        x, y = z.real, z.imag
        fptype = x.dtype.type if isinstance(x, np.ndarray) else type(x)

        one = fptype(1)
        four = fptype(4)
        quarter = fptype(0.25)
        return log1p(x / ((x - one) * (x - one) + y * y) * four) * quarter

    def atanh3(z):
        x, y = z.real, z.imag
        fptype = x.dtype.type if isinstance(x, np.ndarray) else type(x)

        one = fptype(1)
        four = fptype(4)
        quarter = fptype(0.25)
        return -log1p(-x / ((x + one) * (x + one) + y * y) * four) * quarter

    def atanh4(z):
        """
        +inf                                                                 ==
        max                                                                  ==
        8e35                                                                 ==
        2e33                                                               c===
        4e30                                                             c=====
        1e28                                                           c=======
        2e25                                                         c=========
        5e22                                                       c===========
        1e20                                                     c=============
        3e17                                                   c===============
        7e14                                                 c=================
        2e12                                               1===================
        4e9                                              1=====================
        8e6                                            1=======================
        2e4                                          1=========================
        4e1                                        cc==========================
        1e-1                                       cc==========================
        2e-4                                       cc==========================
        5e-7                                       cc==========================
        1e-9                                       cc==========================
        3e-12                                      cc==========================
        6e-15                                      cc==========================
        1e-17                                      cc==========================
        4e-20                                      cc==========================
        8e-23                                      cc==========================
        2e-25                                      cc==========================
        4e-28                                      cc==========================
        1e-30                                      cc==========================
        2e-33                                      cc==========================
        5e-36                                      cc==========================
        tiny                                       cc==========================
        0                                          cc==========================
                0     3e-32 1e-24 5e-17 3e-9  1e-1  5e6   2e14  9e21  4e29  2e37
        """
        x, y = z.real, z.imag
        fptype = x.dtype.type if isinstance(x, np.ndarray) else type(x)

        one = fptype(1)
        four = fptype(4)
        quarter = fptype(0.25)
        return log1p(four / x) * quarter

    def atanh5(z):
        """
        +inf    ==============================================================
        max     ===============================================================
        8e35    ==========================================================1=1==
        2e33    =====================================================11========
        4e30    =================================================1111=====1====
        1e28    =============================================1=11=====1=1111===
        2e25    ===========================================1==1===1==1==11=====
        5e22    =================================================1===111111====
        1e20    ====================================1=11==1=11==1==1=1==11=====
        3e17    ==============================1=1111=====1===1==11===1=1=1==1==
        7e14    ==========================1111111===1=1==21===1===1=====111====
        2e12    ====================1====1===1===1==1==111===11=====1=1==11====
        4e9     ===================11===1====1==1=11===11==1=1===1===1=1=1==1==
        8e6     ===============1=1=1====1===11===11=21=========1==11=1=1=1=====
        2e4     ========1===1==11==11===1=====1=2c cc11===11=1=====11111==1====
        4e1     ====                               cc==1=1=1==1===1=11==1==1===
        1e-1    =                                  cc====1=1==1===111=111=1====
        2e-4    =                                  cc=1=1==1=====11===1===1  ==
        5e-7    =                                  cc===1=1==1=1=1=1=11=1    ==
        1e-9    =                                  cc11=1=1===1==1=1=1       ==
        3e-12   =                                  cc11===11======1=         ==
        6e-15   =                                  cc===1======1=1           ==
        1e-17   =                                  cc===========             ==
        4e-20   =                                  cc11======1               ==
        8e-23   =                                  cc==1111=                 ==
        2e-25   =                                  cc===11                   ==
        4e-28   =                                  cc1=1                     ==
        1e-30   =                                  cc1                       ==
        2e-33   =                                  c                         ==
        5e-36   =                                                            ==
        tiny    =                                                            ==
        0
                0     3e-32 1e-24 5e-17 3e-9  1e-1  5e6   2e14  9e21  4e29  2e37
        """
        x, y = z.real, z.imag
        fptype = x.dtype.type if isinstance(x, np.ndarray) else type(x)

        four = fptype(4)
        quarter = fptype(0.25)
        return log1p(four / (x / y + y / x) / y) * quarter

    def atanh6(z):
        """
        +inf     =============================================================
        max      =============================================================
        8e35     =========================================================   =
        2e33     =====================================================       =
        4e30     =================================================           =
        1e28     ============================================                =
        2e25     ========================================                    =
        5e22     ====================================                        =
        1e20     ================================                            =
        3e17     ============================                                =
        7e14     ========================                                    =
        2e12     ====================                                        =
        4e9      ===============                                             =
        8e6      ===========                                                 =
        2e4      =======                                                     =
        4e1      ===                                                         =
        1e-1     =                                                           =
        2e-4     =                                                           =
        5e-7     1                                                           =
        1e-9     1                                                           =
        3e-12    1                                                           =
        6e-15    1                                                           =
        1e-17    1                                                           =
        4e-20    1                                                           =
        8e-23    1                                                           =
        2e-25    1                                                           =
        4e-28    1                                                           =
        1e-30    1                                                           =
        2e-33    1                                                           =
        5e-36    1                                                           =
        tiny     1                                                           =
        0        1                                                           =
                0     3e-32 1e-24 5e-17 3e-9  1e-1  5e6   2e14  9e21  4e29  2e37
        """
        x, y = z.real, z.imag
        fptype = x.dtype.type if isinstance(x, np.ndarray) else type(x)
        quarter = fptype(0.25)
        return log(x / x) * quarter  # it is effectively zero

    size = 51

    atanh_approx = atanh4

    if 1:
        min_real_value = None
        max_real_value = None
        min_imag_value = None
        max_imag_value = None
    else:
        # study the surrounding of the given point:
        z, dz = -1 - 0.00042062782j, 0.5 + 1e3j
        z, dz = 315387600000000 + 36880000000000j, 1e8 + 1e8j
        min_real_value = z.real - dz.real
        max_real_value = z.real + dz.real
        min_imag_value = z.imag - dz.imag
        max_imag_value = z.imag + dz.imag

    samples = fa.utils.complex_samples(
        (2 * size, size),
        dtype=dtype,
        include_huge=False,
        include_subnormal=False,
        nonnegative=True,
        min_real_value=min_real_value,
        max_real_value=max_real_value,
        min_imag_value=min_imag_value,
        max_imag_value=max_imag_value,
    )[::-1]

    ref = atanh.call(samples).real

    diff = fa.utils.diff_ulp(ref, atanh_approx(samples))

    timage = fa.TextImage()
    timage.fill(0, 10, diff == 0, symbol="=")
    timage.fill(0, 10, diff == 1, symbol="1")
    timage.fill(0, 10, diff == 2, symbol="2")
    timage.fill(0, 10, (diff > 2) & (diff < 100), symbol="c")
    real_axis = samples[0:1].real
    imag_axis = samples[:, 0:1].imag
    timage.insert(0, 2, fa.TextImage.fromseq(imag_axis))
    timage.append(-1, 10, fa.TextImage.fromseq(real_axis[:, ::6], mintextwidth=5, maxtextwidth=5))
    print(timage)

    print(np.finfo(fdtype))


if __name__ == "__main__":
    # main_imag()
    main_real()
