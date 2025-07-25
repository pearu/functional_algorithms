import contextlib
import fractions
import itertools
import numpy
import math
import mpmath
import multiprocessing
import os
import warnings
from collections import defaultdict

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(seq):
        return seq


class _UNSPECIFIED:

    _singleton = None

    def __new__(cls):
        if cls._singleton is None:
            cls._singleton = object.__new__(cls)
        return cls._singleton

    def __str__(self):
        return "UNSPECIFIED"

    __repr__ = __str__


UNSPECIFIED = _UNSPECIFIED()
default_flush_subnormals = False

complex_types = (complex, numpy.complexfloating)
float_types = (float, numpy.floating)
integer_types = (int, numpy.integer)
number_types = (*complex_types, *float_types, *integer_types)
boolean_types = (bool,)
value_types = (*number_types, *boolean_types)


def float2mpf(ctx, x):
    """Convert numpy floating-point number to mpf object."""
    if isinstance(x, numpy.ndarray):
        return [float2mpf(ctx, x_) for x_ in x]

    if numpy.isposinf(x):
        return ctx.make_mpf(mpmath.libmp.finf)
    elif numpy.isneginf(x):
        return ctx.make_mpf(mpmath.libmp.fninf)
    elif numpy.isnan(x):
        return ctx.make_mpf(mpmath.libmp.fnan)
    elif numpy.isfinite(x):
        prec, rounding = ctx._prec_rounding
        mantissa, exponent = numpy.frexp(x)
        man_ = ctx.ldexp(mantissa, prec)
        man = int(man_)
        assert man == man_, (man, man_, x, exponent, prec)
        exp_ = exponent - prec
        exp = int(exp_)
        assert exp == exp_
        r = ctx.make_mpf(mpmath.libmp.from_man_exp(man, exp, prec, rounding))
        assert ctx.isfinite(r), r._mpf_
        return r
    else:
        assert 0  # unreachable


def number2mpf(ctx, x):
    if isinstance(x, list):
        return [number2mpf(ctx, x_) for x_ in x]
    elif isinstance(x, fractions.Fraction):
        return fraction2mpf(ctx, x)
    elif isinstance(x, (numpy.floating, float)):
        return float2mpf(ctx, x)
    elif isinstance(x, int):
        return ctx.mpf(x)
    elif isinstance(x, mpmath.mpf):
        return x
    else:
        assert 0, type(x)  # not impl


def mpf2float(dtype, x, flush_subnormals=False, prec=None, rounding=None):
    """Convert mpf object to numpy floating-point number.

    When x is larger than largest value of the floating point system,
    use round towards zero, otherwise round to nearest.
    """
    if isinstance(x, list):
        return [mpf2float(dtype, x_, flush_subnormals=flush_subnormals, prec=prec, rounding=rounding) for x_ in x]
    ctx = x.context
    if ctx.isfinite(x):
        prec_rounding = ctx._prec_rounding[:]
        if prec is not None:
            prec_rounding[0] = prec
        else:
            prec_rounding[0] = get_precision(dtype)
        if rounding is not None:
            prec_rounding[1] = rounding

        sign, man, exp, bc = mpmath.libmp.normalize(*x._mpf_, *prec_rounding)
        assert bc >= 0, (sign, man, exp, bc, x._mpf_)
        fp_format = dtype.__name__
        zexp = (
            vectorize_with_mpmath.float_minexp[fp_format]
            if flush_subnormals
            else vectorize_with_mpmath.float_subexp[fp_format]
        )
        if exp + bc < zexp:
            return -dtype(0) if sign else dtype(0)
        if exp + bc > vectorize_with_mpmath.float_maxexp[fp_format]:
            return dtype(-numpy.inf) if sign else dtype(numpy.inf)
        # conversion from mpz to longdouble is buggy, see
        # aleaxit/gmpy#507, hence we convert mpz to int:
        man = int(man)
        largest = vectorize_with_mpmath.float_max[fp_format]
        # Two reasons to convert largest to int: (i) numpy longdouble
        # does support comparing against large integers, and (ii)
        # comparison int(man) > dtype(largest) fails with
        # OverflowError.
        largest = int(largest)
        # try avoiding infinity from ldexp(man, exp) by rounding down
        while man > largest:
            man >>= 1
            exp += 1
        r = numpy.ldexp(dtype(man), exp)
        if numpy.isinf(r):
            assert exp > 0
            for e in range(1, vectorize_with_mpmath.float_maxexp[fp_format] - exp):
                m = (man >> e) << e  # round down at the cost of loosing precision
                assert m, (m, man, e, exp)
                r_ = numpy.ldexp(dtype(m), exp)
                if numpy.isfinite(r_):
                    r = r_
                    break
        r = -r if sign else r
        assert numpy.isfinite(r), (x, r, x._mpf_, bc, man, exp, largest, vectorize_with_mpmath.float_maxexp[fp_format])
        return r
    elif ctx.isnan(x):
        return dtype(numpy.nan)
    elif ctx.isinf(x):
        return dtype(-numpy.inf if x._mpf_[0] else numpy.inf)
    else:
        assert 0  # unreachable


def tobinary(x):
    """Return binary representation of a floating-point number instance."""
    if isinstance(x, numpy.floating):
        return float2bin(x)
    elif isinstance(x, mpmath.mpf):
        return mpf2bin(x)
    else:
        raise NotImplementedError(f"tobinary({type(x)})")


def mpf2bin(s):
    s = s._mpf_
    if s == mpmath.libmp.libmpf.fzero:
        return "0"
    elif s == mpmath.libmp.libmpf.finf:
        return "inf"
    elif s == mpmath.libmp.libmpf.fninf:
        return "-inf"
    elif s == mpmath.libmp.libmpf.fnan:
        return "nan"
    sign = "-" if s[0] else ""
    prec = s[1].bit_length()
    s = mpmath.libmp.libmpf.normalize(*s, prec, mpmath.libmp.libmpf.round_nearest)
    digits = bin(s[1])[2:]
    if len(digits) > 1:
        digits = digits[0] + "." + digits[1:]
    exponent = s[2]
    if s[1]:
        exponent += s[1].bit_length() - 1
    return f"{sign}{digits}p{exponent:+01d}"


def mpfnextafter(x, y):
    """Return next mpf value to x in the direction of y.

    Warning: the concept of subnormals and infinities are not
    supported because the values of mpf exponents are unlimited.
    """
    s, m, e, b = x._mpf_
    prec = x.context._prec_rounding[0]
    assert b <= prec, (x._mpf_, prec)
    m1, e1 = m * (2 ** (prec - b)), e - (prec - b)
    if x == y:
        r = x
    elif (x > y and x < 0) or (x < y and x > 0):
        r = x.context.make_mpf((s, m1 + 1, e1, prec))
    elif x == 0:
        if y > 0:
            r = x.context.make_mpf((0, 1, -prec - 1, 1))
        else:
            r = x.context.make_mpf((1, 1, -prec - 1, 1))
    else:
        if m1 <= 2 ** (prec - 1):
            m1 *= 2
            e1 -= 1
        r = x.context.make_mpf((s, m1 - 1, e1, prec))
    return r


def matching_bits(x, y):
    """Return the number of matching bits of floating point numbers.

    The highest return value (the number of significant bits) is
    returned when x == y.

    The zero return value indicates that x and y have the same order
    of magnitude.

    The negative return value indicates that x and y are different
    either in sign or magnitude.

    For fixed x and monotonically varying y, matching_bits(x, y) is a
    concave function.

    """
    if isinstance(x, numpy.ndarray) and isinstance(y, numpy.ndarray):
        assert len(x) == len(y)
        lst = []
        for x_, y_ in zip(x, y):
            lst.append(matching_bits(x_, y_))
        return numpy.array(lst)

    if isinstance(x, list) and isinstance(y, list):
        assert len(x) == len(y)
        lst = []
        for x_, y_ in zip(x, y):
            lst.append(matching_bits(x_, y_))
        return lst

    if isinstance(x, mpmath.mp.mpf) and isinstance(y, mpmath.mp.mpf):

        ctx = x.context
        prec = ctx._prec_rounding[0]

        if x == y:
            return prec
        elif x < y:
            return matching_bits(y, x)
        elif ctx.isinf(x) or ctx.isnan(x):
            return -2
        elif y.context.isinf(y) or y.context.isnan(y):
            return -2
        # normalize to context precision
        x = x + 0
        y = y + 0

        xsign, xman, xexp, xbc = x._mpf_
        ysign, yman, yexp, ybc = y._mpf_
        # x, y -> man1 * 2 ** exp, man2 * 2 ** exp
        if xexp < yexp:
            man1 = [1, -1][xsign] * xman
            man2 = [1, -1][ysign] * yman * (2 ** (yexp - xexp) if yman else 1)
        else:
            man1 = [1, -1][xsign] * xman * (2 ** (xexp - yexp) if xman else 1)
            man2 = [1, -1][ysign] * yman
        r = round(prec * (1 - (man1 - man2).bit_length() / max(1, min(man1.bit_length(), man2.bit_length()))), 1)
        return max(r, -1)
    elif isinstance(x, numpy.floating) and isinstance(y, type(x)):
        dtype = type(x)
        fi = numpy.finfo(dtype)
        prec = abs(fi.negep)
        if x == y:
            return prec
        elif x < y:
            return matching_bits(y, x)
        elif numpy.isinf(x):
            return -2
        elif numpy.isinf(y):
            return -2
        xf, xe = numpy.frexp(x)
        yf, ye = numpy.frexp(y)
        if xe == ye:
            f1, f2 = xf, yf
        elif xe < ye:
            f1 = xf
            f2 = yf * 2 ** (ye - xe)
        else:
            f1 = xf * 2 ** (xe - ye)
            f2 = yf
        df, de = numpy.frexp(abs(f1 - f2))
        return max(-de, -1)
    else:
        raise NotImplementedError(f"matching digits of {type(x).__name__} and {type(y).__name__}")


def diff_prec(x, y):
    """Return"""
    if isinstance(x, mpmath.mp.mpf) and isinstance(y, mpmath.mp.mpf):
        if x == y:
            return x.context._prec_rounding[0]
        d = x - y
        xsign, xman, xexp, xbc = x._mpf_
        _, dman, dexp, _ = d._mpf_
        xe = xexp + xman.bit_length() - 1
        de = dexp + dman.bit_length() - 1
        return xe - de
    elif isinstance(x, numpy.floating) and isinstance(y, type(x)):
        if x == y:
            fi = numpy.finfo(type(x))
            return abs(fi.machep)
        xf, xe = numpy.frexp(x)
        df, de = numpy.frexp(x - y)
        return xe - de
    else:
        raise NotImplementedError(f"matching digits of {type(x).__name__} and {type(y).__name__}")


def bin2float(dtype, b):
    if b == "0":
        return dtype(0)
    elif b == "-inf":
        return -dtype(numpy.inf)
    elif b == "inf":
        return dtype(numpy.inf)
    elif b == "nan":
        return dtype(numpy.nan)

    exponent_width, significant_width, uint = {
        numpy.float16: (5, 10, numpy.uint16),
        numpy.float32: (8, 23, numpy.uint32),
        numpy.float64: (11, 52, numpy.uint64),
    }[dtype]

    isneg = b.startswith("-")
    if isneg:
        b = b[1:]

    i = b.index("p")
    e = int(b[i + 1 :]) + (1 << (exponent_width - 1)) - 1
    b = b[:i]

    if b.startswith("1."):
        significant_bits = b[2:]
    elif b == "1":
        significant_bits = "0"
    else:
        assert 0, b  # unreachable

    s = significant_width - len(significant_bits)
    if e <= 0:
        significant_bits = "1" + significant_bits
        s -= 1

    significant_bits = uint(int(significant_bits, 2))
    significant_bits <<= uint(s)

    if e <= 0:
        ivalue = significant_bits >> uint(-e)
    else:
        ivalue = significant_bits + uint(e << significant_width)

    value = ivalue.view(dtype)
    if isneg:
        value = -value

    return value


def float2bin(f):

    total_bits, exponent_width, integer_part_width, significant_width, uint = {
        numpy.float16: (16, 5, 0, 10, numpy.uint16),
        numpy.float32: (32, 8, 0, 23, numpy.uint32),
        numpy.float64: (64, 11, 0, 52, numpy.uint64),
        numpy.longdouble: (128, 15, 1, 63, None),
    }[type(f)]

    def get_digits(x):
        if isinstance(x, numpy.longdouble):
            x1, x2 = numpy.frombuffer(x.tobytes(), dtype=numpy.uint64)
            return bin(x1)[2:] + bin(x2)[2:]
        return bin(x.view(uint))[2:]

    digits = get_digits(f)
    if f >= 0:
        sign = ""
        digits = "0" * (total_bits - len(digits)) + digits
    elif f < 0:
        sign = "-"
        digits = digits + "0" * (total_bits - len(digits))
    else:
        # nan
        return "nan"
    assert len(digits) == total_bits, (len(digits), total_bits, f)

    sign_width = 1
    exponent_bits = digits[sign_width : sign_width + exponent_width]
    significant_bits = digits[sign_width + exponent_width + integer_part_width :].rstrip("0")

    eb = 2 ** (exponent_width - 1)
    e = int(exponent_bits, 2) - eb + 1
    if e == eb:
        assert significant_bits == "", (significant_bits, digits)
        return f"{sign}inf"
    elif e == -eb + 1:
        if not significant_bits:
            return "0"
        # subnormal has no leading significant
        k = len(significant_bits)
        significant_bits = significant_bits.lstrip("0")
        e -= k - len(significant_bits)
        if significant_bits:
            significant_bits = significant_bits[1:]

    if significant_bits:
        return f"{sign}1.{significant_bits}p{e:+01d}"
    return f"{sign}1p{e:+01d}"


def float2fraction(f):
    """Convert floating-point number to Fraction.
    The conversion is exact.
    """
    if isinstance(f, list):
        return sum(map(float2fraction, f))
    elif isinstance(f, numpy.floating):
        dtype = type(f)
        fi = numpy.finfo(dtype)
        itype = {numpy.float16: numpy.uint16, numpy.float32: numpy.uint32, numpy.float64: numpy.uint64}[dtype]
        i = f.view(itype)
        one = itype(1)
        ssz = 1  # bit-size of sign part
        esz = itype(fi.nexp)  # bit-size of exponential part
        fsz = itype(-ssz - fi.negep)  # bit-size of fractional part
        fmask = itype((one << fsz) - one)
        emask = itype((one << esz) - one)
        umask = itype((one << (esz + fsz)) - one)
        mxu = int(one << fsz)

        u = i & umask

        fpart = int(u & fmask)
        epart = int((u >> fsz) & emask)
        s = 1 if f < 0 else 0
        e = epart + fi.minexp - 1

        if epart == 0 and fpart == 0:
            # signed zero
            num = 0
            denom = 1 - 2 * s
        elif epart == 0:
            # subnormals
            num = (1 - 2 * s) * (fpart)
            denom = mxu * (1 << (-e - 1))
        elif epart == emask and fpart == 0:
            # infinity
            num = (1 - 2 * s) * (1 << fi.maxexp)
            denom = 1
        elif e < 0:
            num = (1 - 2 * s) * (mxu + fpart)
            denom = mxu * (1 << (-e))
        else:
            num = (1 - 2 * s) * (mxu + fpart) * (1 << e)
            denom = mxu
        return fractions.Fraction(num, denom)
    elif isinstance(f, float):
        return float2fraction(numpy.float64(f))
    elif isinstance(f, mpmath.mpf):
        sign, man, exp, bc = f._mpf_
        if man == 0 and exp == 0:
            num = (1 - 2 * sign) * 0
            denom = 1
        elif man == 0:
            # infinity
            k = 1
            while 2**k < f.context.prec:
                k += 1
            maxexp = 2 ** (3 * k - 8)  # k=4->16, 5->128, 6->1024
            num = (1 - 2 * sign) * (2**maxexp)
            denom = 1
        elif exp < 0:
            num = man * (1 - 2 * sign)
            denom = 2 ** (-exp)
        else:
            num = man * 2 ** (exp) * (1 - 2 * sign)
            denom = 1
        return fractions.Fraction(num, denom)
    raise TypeError(f"float to fraction conversion requires floating-point input, got {type(f).__name__}")


def number2fraction(x):
    if isinstance(x, list):
        return list(map(number2fraction, x))
    elif isinstance(x, (mpmath.mpf, float, numpy.floating)):
        return float2fraction(x)
    elif isinstance(x, int):
        return fractions.Fraction(x, 1)
    elif isinstance(x, fractions.Fraction):
        return x
    raise TypeError(f"float to fraction conversion requires int|floating-point|mpf input, got {type(x).__name__}")


def fraction2float(dtype, q):
    """Convert Fraction to a floating-point number.
    The conversion may be inexact.
    """
    fi = numpy.finfo(dtype)
    num, denom = q.numerator, q.denominator
    unum = abs(num)
    if num == 0:
        return dtype(0) if denom == 1 else -dtype(0)
    elif denom == 1 and unum >= (1 << fi.maxexp):
        return -dtype(numpy.inf) if num < 0 else dtype(numpy.inf)
    else:
        try:
            f = num / denom
            if abs(f) > fi.max:
                raise OverflowError
            return dtype(f)
        except OverflowError:
            return mpf2float(dtype, mpmath.mpf(num) / denom)


def fraction2mpf(ctx, q):
    num, denom = q.numerator, q.denominator
    return ctx.mpf(num) / denom


def number2float(dtype, q):
    if isinstance(q, list):
        return list(number2float(dtype, q_) for q_ in q)
    elif isinstance(q, fractions.Fraction):
        return fraction2float(dtype, q)
    elif isinstance(q, (numpy.floating, float, int)):
        return dtype(q)
    elif isinstance(q, mpmath.mpf):
        return mpf2float(dtype, q)
    else:
        assert 0, (dtype, type(q))  # not impl


def get_precision(x):
    """Return significant precision of x in bits."""
    if isinstance(x, numpy.ndarray):
        x = x.dtype.type
    elif isinstance(x, numpy.floating):
        x = type(x)
    return -numpy.finfo(x).negep


def get_maxexp(x):
    """Return maximal binary exponent of x."""
    if isinstance(x, numpy.ndarray):
        x = x.dtype.type
    elif isinstance(x, numpy.floating):
        x = type(x)
    return numpy.finfo(x).maxexp


def get_veltkamp_splitter_constant(x):
    p = get_precision(x)
    s = (p + 1) // 2
    if isinstance(x, numpy.floating):
        x = type(x)
    return x(2**s + 1)


def split_veltkamp_max(x):
    """Return maximal s such that

      (2 ** s + 1) * x

    is finite.
    """
    _, e = numpy.frexp(x)
    p = int(-numpy.finfo(type(x)).machep)
    s = min(int(numpy.finfo(type(x)).maxexp - e + 2), p - 2)

    # TODO: find s without requiring the while loop
    while s < p - 2 and numpy.isfinite(type(x)(2 ** (s + 1) + 1) * x):
        s += 1
    return s


def split_veltkamp(x, s=None, C=None):
    """Return xh and xl such that

      x = xh + xl

    where xh fits into p-s bits, xl fits into s bits, and p is the
    precision of floating point system.
    """
    # https://inria.hal.science/hal-04480440v1
    # https://hal.science/hal-04624238/document
    if C is None:
        p = get_precision(x)
        if s is None:
            s = (p + 1) // 2
        assert s >= 2 and s <= p - 2
        C = type(x)(2**s + 1)
    g = C * x  # for large x and s, this will overflow!
    d = x - g
    xh = g + d
    xl = x - xh
    return xh, xl


def multiply_dekker(x, y, C=None):
    """Dekker's product.

    Returns r1, r2 such that

      x * y == r1 + r2
    """
    if C is None:
        p = get_precision(x)
        s = (p + 1) // 2
        assert type(x) is type(y)
        xh, xl = split_veltkamp(x, s)
        yh, yl = split_veltkamp(y, s)
    else:
        xh, xl = split_veltkamp(x, C=C)
        yh, yl = split_veltkamp(y, C=C)
    r1 = x * y
    t1 = (-r1) + xh * yh
    t2 = t1 + xh * yl
    t3 = t2 + xl * yh
    r2 = t3 + xl * yl
    return r1, r2


def square_dekker(x, C=None):
    """Square using Dekker's product.

    Returns r1, r2 such that

      x * x == r1 + r2
    """
    if C is None:
        p = get_precision(x)
        s = (p + 1) // 2
        xh, xl = split_veltkamp(x, s)
    else:
        xh, xl = split_veltkamp(x, C=C)
    r1 = x * x
    t1 = (-r1) + xh * xh
    t2 = t1 + xh * xl
    t3 = t2 + xl * xh
    r2 = t3 + xl * xl
    return r1, r2


def add_2sum(x, y):
    """Sum of x and y.

    Return s, t such that

      x + y = s + t
    """
    s = x + y
    z = s - x
    t = (x - (s - z)) + (y - z)
    return s, t


def add_fast2sum(x, y):
    """Sum of x and y where abs(x) >= abs(y)

    Return s, t such that

      x + y = s + t
    """
    s = x + y
    z = s - x
    t = y - z
    return s, t


def sum_2sum(seq):
    """Sum all items in a sequence using 2Sum algorithm."""
    if len(seq) == 1:
        return seq[0], type(seq[0])(0)
    elif len(seq) == 2:
        return add_2sum(*seq)
    elif len(seq) >= 3:
        s, t = add_2sum(*seq[:2])
        for n in seq[2:]:
            s, t1 = add_2sum(s, n)
            t = t + t1
        return add_2sum(s, t)
    assert 0  # unreachable


def sum_fast2sum(seq):
    """Sum all items in a sequence using Fast2Sum algorithm."""
    if len(seq) == 1:
        return seq[0], type(seq[0])(0)
    elif len(seq) == 2:
        return add_fast2sum(*seq)
    elif len(seq) >= 3:
        s, t = add_fast2sum(*seq[:2])
        for n in seq[2:]:
            s, t1 = add_fast2sum(s, n)
            t = t + t1
        return add_fast2sum(s, t)
    assert 0  # unreachable


def double_2sum(x):
    """
    Return s, t such that

      x + x = s + t
    """
    s = x + x
    z = s - x
    t = (x - (s - z)) + (x - z)
    return s, t


def double_fast2sum(x):
    """
    Return s, t such that

      x + x = s + t
    """
    s = x + x
    z = s - x
    t = x - z
    return s, t


class vectorize_with_backend(numpy.vectorize):

    pyfunc_is_vectorized = False

    @classmethod
    def backend_is_available(cls, device):
        return device == "cpu"

    def __init__(self, *args, **kwargs):
        self.device = kwargs.pop("device", "cpu")

        kwargs.pop("dtype", None)
        super().__init__(*args, **kwargs)

    @property
    def backend_types(self):
        raise NotImplementedError(f"{type(self).__name__}.backend_types")

    def backend_context(self, context):
        raise NotImplementedError(f"{type(self).__name__}.backend_context")

    def numpy_to_backend(self, arr):
        raise NotImplementedError(f"{type(self).__name__}.numpy_to_backend")

    def numpy_from_backend(self, obj):
        raise NotImplementedError(f"{type(self).__name__}.numpy_from_backend")

    def context_from_backend(self, obj):
        raise NotImplementedError(f"{type(self).__name__}.context_from_backend")

    def _call_eval(self, sample):
        context = self.context_from_backend(sample)
        sample = self.numpy_to_backend(sample)
        with self.backend_context(context):
            with warnings.catch_warnings(action="ignore"):
                if isinstance(sample, tuple):
                    result = super().__call__(*sample)
                else:
                    result = super().__call__(sample)
        return self.numpy_from_backend(result)

    def call(self, samples, workers=None, enable_progressbar=False):
        """Apply function to samples using concurrency with the given number
        of workers. When workers is None, use os.cpu_count() workers.

        Using this method is unnecessary if the backend already
        supports multithreading functions evaluations.
        """
        if workers is None:
            workers = min(os.cpu_count(), len(samples))

        if enable_progressbar:

            def progress(iter, total):
                yield from tqdm(iter, total=total)

        else:

            def progress(iter, total):
                return iter

        results = []
        if workers > 1:
            # cannot use the default `fork` start method with JAX:
            ctx = multiprocessing.get_context("spawn" if isinstance(self, vectorize_with_jax) else "fork")
            with ctx.Pool(workers) as p:
                for result in p.starmap(self._call_eval, [(s,) for s in samples]):
                    results.append(result)
        else:
            for sample in progress(samples, total=len(samples)):
                results.append(self._call_eval(sample))
        if results and isinstance(results[0], (numpy.number, numpy.ndarray)):
            return numpy.array(results)
        return results

    def __call__(self, *args, **kwargs):
        mp_args = []
        context = None
        for a in args:
            if isinstance(a, (numpy.ndarray, numpy.floating, numpy.complexfloating)):
                if context is None:
                    context = self.context_from_backend(a)  # to be used as a __call__ context below
                with self.backend_context(self.context_from_backend(a)):
                    a = self.numpy_to_backend(a)
                assert self.device.upper() in str(getattr(a, "device", "cpu")).upper()
            mp_args.append(a)

        with warnings.catch_warnings(action="ignore"):
            with self.backend_context(context):
                if self.pyfunc_is_vectorized:
                    result = self.pyfunc(*mp_args, **kwargs)
                else:
                    result = super().__call__(*mp_args, **kwargs)

        if isinstance(result, tuple):
            lst = []
            for r in result:
                if (isinstance(r, numpy.ndarray) and r.dtype.kind == "O") or isinstance(r, self.backend_types):
                    r = self.numpy_from_backend(r)
                lst.append(r)
            return tuple(lst)

        if (isinstance(result, numpy.ndarray) and result.dtype.kind == "O") or isinstance(result, self.backend_types):
            return self.numpy_from_backend(result)

        return result


class vectorize_with_mpmath(vectorize_with_backend):
    """Same as numpy.vectorize but using mpmath backend for function evaluation."""

    map_float_to_complex = dict(
        float16="complex32", float32="complex64", float64="complex128", float128="complex256", longdouble="clongdouble"
    )
    map_complex_to_float = {v: k for k, v in map_float_to_complex.items()}

    float_prec = dict(float16=11, float32=24, float64=53, float128=113, longdouble=64)

    float_subexp = dict(float16=-23, float32=-148, float64=-1073, float128=-16444, longdouble=-16444)
    float_minexp = dict(float16=-13, float32=-125, float64=-1021, float128=-16381, longdouble=-16381)

    float_maxexp = dict(
        float16=16,
        float32=128,
        float64=1024,
        float128=16384,
        longdouble=16384,
    )

    float_max = dict(
        float16=numpy.nextafter(numpy.float16(numpy.inf), numpy.float16(0)),
        float32=numpy.nextafter(numpy.float32(numpy.inf), numpy.float32(0)),
        float64=numpy.nextafter(numpy.float64(numpy.inf), numpy.float64(0)),
        float128=numpy.nextafter(numpy.float128(numpy.inf), numpy.float128(0)),
        longdouble=numpy.nextafter(numpy.longdouble(numpy.inf), numpy.longdouble(0)),
    )

    @classmethod
    def backend_is_available(cls, device):
        return device == "cpu"

    def __init__(self, *args, **kwargs):
        self.extra_prec_multiplier = kwargs.pop("extra_prec_multiplier", 0)
        self.extra_prec = kwargs.pop("extra_prec", 0)
        flush_subnormals = kwargs.pop("flush_subnormals", UNSPECIFIED)
        self.flush_subnormals = flush_subnormals if flush_subnormals is UNSPECIFIED else default_flush_subnormals
        self._contexts = None
        self._contexts_inv = None
        super().__init__(*args, **kwargs)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_contexts"]
        del state["_contexts_inv"]
        return state

    def __setstate__(self, state):
        self._contexts = None
        self._contexts_inv = None
        self.__dict__.update(state)

    def _make_contexts(self):
        contexts = dict()
        contexts_inv = dict()
        for fp_format, prec in self.float_prec.items():
            ctx = mpmath.mp.clone()
            ctx.prec = prec
            contexts[fp_format] = ctx
            contexts_inv[ctx] = fp_format
        return contexts, contexts_inv

    @property
    def contexts(self):
        if self._contexts is None:
            self._contexts, self._contexts_inv = self._make_contexts()
        return self._contexts

    @property
    def contexts_inv(self):
        if self._contexts_inv is None:
            self._contexts, self._contexts_inv = self._make_contexts()
        return self._contexts_inv

    def get_context(self, x):
        if isinstance(x, tuple):
            for x_ in x:
                if isinstance(x_, (numpy.ndarray, numpy.floating, numpy.complexfloating)):
                    return self.get_context(x_)
        if isinstance(x, (numpy.ndarray, numpy.floating, numpy.complexfloating)):
            fp_format = str(x.dtype)
            fp_format = self.map_complex_to_float.get(fp_format, fp_format)
            return self.contexts[fp_format]
        raise NotImplementedError(f"get mpmath context from {type(x).__name__} instance")

    def nptomp(self, x):
        """Convert numpy array/scalar to an array/instance of mpmath number type."""
        if isinstance(x, tuple):
            return tuple(map(self.nptomp, x))
        elif isinstance(x, numpy.ndarray):
            return numpy.fromiter(map(self.nptomp, x.flatten()), dtype=object).reshape(x.shape)
        elif isinstance(x, numpy.floating):
            ctx = self.get_context(x)
            return float2mpf(ctx, x)
        elif isinstance(x, numpy.complexfloating):
            re, im = self.nptomp(x.real), self.nptomp(x.imag)
            return re.context.make_mpc((re._mpf_, im._mpf_))
        elif isinstance(x, (str, bool, int, float, complex, dict, list, set)):
            return x
        raise NotImplementedError(f"convert {type(x).__name__} instance to mpmath number type")

    def mptonp(self, x):
        """Convert mpmath instance to numpy array/scalar type."""
        if isinstance(x, tuple):
            return tuple(map(self.mptonp, x))
        elif isinstance(x, numpy.ndarray) and x.dtype.kind == "O":
            x_flat = x.flatten()
            item = x_flat[0]
            ctx = item.context
            fp_format = self.contexts_inv[ctx]
            if isinstance(item, ctx.mpc):
                dtype = getattr(numpy, self.map_float_to_complex[fp_format])
            elif isinstance(item, ctx.mpf):
                dtype = getattr(numpy, fp_format)
            else:
                dtype = None
            if dtype is not None:
                return numpy.fromiter(map(self.mptonp, x_flat), dtype=dtype).reshape(x.shape)
        elif isinstance(x, mpmath.ctx_mp.mpnumeric):
            ctx = x.context
            if isinstance(x, ctx.mpc):
                fp_format = self.contexts_inv[ctx]
                dtype = getattr(numpy, self.map_float_to_complex[fp_format])
                r = dtype().reshape(1).view(getattr(numpy, fp_format))
                r[0] = self.mptonp(x.real)
                r[1] = self.mptonp(x.imag)
                return r.view(dtype)[0]
            elif isinstance(x, ctx.mpf):
                fp_format = self.contexts_inv[ctx]
                dtype = getattr(numpy, fp_format)
                return mpf2float(dtype, x, flush_subnormals=self.flush_subnormals)
        elif isinstance(x, (str, bool, int, float, complex, dict, list, set)):
            return x

        raise NotImplementedError(f"convert {type(x)} instance to numpy floating point type")

    def context_from_backend(self, sample):
        return self.get_context(sample)

    def backend_context(self, context):
        extraprec = int(context.prec * self.extra_prec_multiplier) + self.extra_prec
        return context.extraprec(extraprec)

    def numpy_to_backend(self, arr):
        return self.nptomp(arr)

    def numpy_from_backend(self, obj):
        return self.mptonp(obj)

    @property
    def backend_types(self):
        return (mpmath.ctx_mp.mpnumeric,)


class vectorize_with_jax(vectorize_with_backend):

    pyfunc_is_vectorized = True

    @classmethod
    def backend_is_available(cls, device):
        import jax

        try:
            return bool(jax.devices(device))
        except RuntimeError:
            return False

    def backend_context(self, context):
        return context

    def numpy_to_backend(self, arr):
        import jax

        return jax.numpy.array(arr)

    def numpy_from_backend(self, obj):
        return numpy.array(obj)

    @property
    def backend_types(self):
        import jaxlib

        return (jaxlib.xla_extension.ArrayImpl,)

    def context_from_backend(self, sample):
        import jax

        jax.config.update("jax_enable_x64", sample.dtype.type in {numpy.float64, numpy.complex128})
        return jax.default_device(jax.devices(self.device)[0])


class vectorize_with_numpy(vectorize_with_backend):

    def backend_context(self, context):
        return contextlib.nullcontext()

    def context_from_backend(self, obj):
        return contextlib.nullcontext()

    def numpy_to_backend(self, arr):
        return arr

    def numpy_from_backend(self, obj):
        assert isinstance(obj, numpy.ndarray), type(obj)
        return obj

    @property
    def backend_types(self):
        return (numpy.ndarray,)


class mpmath_array_api:
    """Array API interface to mpmath functions including workarounds to
    mpmath bugs.
    """

    def hypot(self, x, y):
        return x.context.hypot(x, y)

    def abs(self, x):
        return x.context.absmin(x)

    def absolute(self, x):
        return x.context.absmin(x)

    def exp(self, x):
        return x.context.exp(x)

    def sin(self, x):
        return x.context.sin(x)

    def cos(self, x):
        return x.context.cos(x)

    def sinh(self, x):
        return x.context.sinh(x)

    def cosh(self, x):
        return x.context.cosh(x)

    def conjugate(self, x):
        return x.context.conjugate(x)

    conj = conjugate

    def sign(self, x):
        return x.context.sign(x)

    def sinc(self, x):
        return x.context.sinc(x)

    def positive(self, x):
        return x

    def negative(self, x):
        return -x

    def square(self, x):
        ctx = x.context
        if isinstance(x, ctx.mpc):
            if ctx.isfinite(x.real) and abs(x.real) == abs(x.imag):
                return ctx.make_mpc((ctx.zero._mpf_, (x.real * x.imag * 2)._mpf_))
            return ctx.make_mpc((((x.real - x.imag) * (x.real + x.imag))._mpf_, (x.real * x.imag * 2)._mpf_))
        return x * x

    def expm1(self, x):
        ctx = x.context
        return ctx.expm1(x)

    def _log_at_inf(self, re, im):
        # Workaround mpmath 1.3 bug in log(+-inf+-infj) evaluation (see mpmath/mpmath#774).
        ctx = re.context
        pi = ctx.pi
        if re > 0 and im > 0:
            return ctx.make_mpc((re._mpf_, (pi / 4)._mpf_))
        elif re > 0 and im < 0:
            return ctx.make_mpc((re._mpf_, (-pi / 4)._mpf_))
        elif re < 0 and im < 0:
            return ctx.make_mpc(((-re)._mpf_, (-3 * pi / 4)._mpf_))
        elif re < 0 and im > 0:
            return ctx.make_mpc(((-re)._mpf_, (3 * pi / 4)._mpf_))
        else:
            assert 0  # unreachable

    def log1p(self, x):
        ctx = x.context
        if isinstance(x, ctx.mpc):
            if ctx.isinf(x.real) and ctx.isinf(x.imag):
                return self._log_at_inf(x.real, x.imag)
        else:
            if x < -1:
                # otherwise, mpmath.log1p returns a complex value
                return ctx.nan

        r = ctx.log1p(x)
        if isinstance(x, ctx.mpc):
            if isinstance(r, ctx.mpf):
                # Workaround log1p(0j) -> 0
                r = ctx.make_mpc((r._mpf_, ctx.zero._mpf_))
        return r

    def log(self, x):
        ctx = x.context
        if isinstance(x, ctx.mpc):
            if ctx.isinf(x.real) and ctx.isinf(x.imag):
                return self._log_at_inf(x.real, x.imag)
        else:
            if x < 0:
                # otherwise, mpmath.log returns a complex value
                return ctx.nan
        return ctx.log(x)

    def tan(self, x):
        ctx = x.context
        if isinstance(x, ctx.mpc):
            if not (ctx.isfinite(x.real) and ctx.isfinite(x.imag)):
                # tan(z) = -i * std::tanh(i * z)
                ix = ctx.make_mpc(((-x.imag)._mpf_, x.real._mpf_))
                w = self.tanh(ix)
                return ctx.make_mpc((w.imag._mpf_, (-w.real)._mpf_))
        return ctx.tan(x)

    def tanh(self, x):
        ctx = x.context
        if isinstance(x, ctx.mpc):
            # Workaround mpmath 1.3 bug in tanh(+-inf+-infj) evaluation (see mpmath/mpmath#781).
            if ctx.isfinite(x.real) and not ctx.isfinite(x.imag):
                if x.real == 0:
                    return ctx.make_mpc((ctx.zero._mpf_, ctx.nan._mpf_))
                return ctx.make_mpc((ctx.nan._mpf_, ctx.nan._mpf_))
            elif ctx.isinf(x.real):
                if x.real >= 0:
                    return ctx.make_mpc((ctx.one._mpf_, ctx.zero._mpf_))
                return ctx.make_mpc(((-ctx.one)._mpf_, ctx.zero._mpf_))
            elif ctx.isnan(x.real):
                if x.imag == 0:
                    return ctx.make_mpc((ctx.nan._mpf_, ctx.zero._mpf_))
                return ctx.make_mpc((ctx.nan._mpf_, ctx.nan._mpf_))

        return ctx.tanh(x)

    def log2(self, x):
        return self.log(x) / x.context.ln2

    def log10(self, x):
        return self.log(x) / x.context.ln10

    def exp2(self, x):
        return x.context.exp(x * x.context.ln2)

    def arcsin(self, x):
        ctx = x.context
        if isinstance(x, ctx.mpc):
            # Workaround mpmath 1.3 bug in asin(+-inf+-infj) evaluation (see
            # mpmath/mpmath#793).
            pi = ctx.pi
            inf = ctx.inf
            zero = ctx.zero
            if ctx.isinf(x.real):
                sign_real = -1 if x.real < 0 else 1
                real = sign_real * pi / (4 if ctx.isinf(x.imag) else 2)
                imag = -inf if x.imag < 0 else inf
                return ctx.make_mpc((real._mpf_, imag._mpf_))
            elif ctx.isinf(x.imag):
                return ctx.make_mpc((zero._mpf_, x.imag._mpf_))
            # On branch cut, mpmath.mp.asin returns different value
            # compared to mpmath.fp.asin and numpy.arcsin. The
            # following if-block ensures compatibiliy with
            # numpy.arcsin.
            if x.real > 1 and x.imag == 0:
                return ctx.asin(x).conjugate()
        else:
            if abs(x) > 1:
                # otherwise, mpmath.asin would return complex value
                return ctx.nan
        return ctx.asin(x)

    def arcsinh(self, x):
        ctx = x.context

        if isinstance(x, ctx.mpc):
            # Workaround mpmath 1.3 bug in asinh(+-inf+-infj) evaluation
            # (see mpmath/mpmath#749).
            pi = ctx.pi
            inf = ctx.inf
            zero = ctx.zero
            if ctx.isinf(x.imag):
                sign_imag = -1 if x.imag < 0 else 1
                real = -inf if x.real < 0 else inf
                imag = sign_imag * pi / (4 if ctx.isinf(x.real) else 2)
                return ctx.make_mpc((real._mpf_, imag._mpf_))
            elif ctx.isinf(x.real):
                return ctx.make_mpc((x.real._mpf_, zero._mpf_))
            # On branch cut, mpmath.mp.asinh returns different value
            # compared to mpmath.fp.asinh and numpy.arcsinh (see
            # mpmath/mpmath#786).  The following if-block ensures
            # compatibiliy with numpy.arcsinh.
            if x.real == 0 and x.imag < -1:
                return (-ctx.asinh(x)).conjugate()
        return ctx.asinh(x)

    def arccos(self, x):
        ctx = x.context
        if isinstance(x, ctx.mpc):
            # Workaround mpmath 1.3 bug in acos(+-inf+-infj) evaluation (see
            # mpmath/mpmath#795 for a fix).
            pi = ctx.pi
            inf = ctx.inf
            zero = ctx.zero
            if ctx.isinf(x.imag):
                if ctx.isinf(x.real):
                    real = pi / 4 if x.real > 0 else 3 * pi / 4
                else:
                    real = pi / 2
                imag = inf if x.imag < 0 else -inf
                return ctx.make_mpc((real._mpf_, imag._mpf_))
            elif ctx.isinf(x.real):
                inf = ctx.inf
                sign_imag = -1 if x.imag < 0 else 1
                real = zero if x.real > 0 else pi
                return ctx.make_mpc((real._mpf_, (-sign_imag * inf)._mpf_))
            # On branch cut, mpmath.mp.acos returns different value
            # compared to mpmath.fp.acos and numpy.arccos. The
            # following if-block ensures compatibiliy with
            # numpy.arccos.
            if x.imag == 0 and x.real > 1:
                return -ctx.acos(x)
        else:
            if abs(x) > 1:
                # otherwise, mpmath.asin would return complex value
                return ctx.nan
        return ctx.acos(x)

    def arccosh(self, x):
        ctx = x.context

        if isinstance(x, ctx.mpc):
            # Workaround mpmath 1.3 bug in acosh(+-inf+-infj) evaluation
            # (see mpmath/mpmath#749).
            pi = ctx.pi
            inf = ctx.inf
            zero = ctx.zero
            if ctx.isinf(x.real):
                sign_imag = -1 if x.imag < 0 else 1
                imag = (
                    (3 if x.real < 0 else 1) * sign_imag * pi / 4
                    if ctx.isinf(x.imag)
                    else (sign_imag * pi if x.real < 0 else zero)
                )
                return ctx.make_mpc((inf._mpf_, imag._mpf_))
            elif ctx.isinf(x.imag):
                sign_imag = -1 if x.imag < 0 else 1
                imag = sign_imag * pi / 2
                return ctx.make_mpc((inf._mpf_, imag._mpf_))
        else:
            if x < 1:
                # otherwise, mpmath.acosh would return complex value
                return ctx.nan
        return ctx.acosh(x)

    def sqrt(self, x):
        ctx = x.context
        if isinstance(x, ctx.mpc):
            # Workaround mpmath 1.3 bug in sqrt(a+-infj) evaluation
            # (see mpmath/mpmath#776)
            if ctx.isinf(x.imag):
                return ctx.make_mpc((ctx.inf._mpf_, x.imag._mpf_))
        else:
            if x < 0:
                # otherwise, mpmath.sqrt would return complex value
                return ctx.nan
        return ctx.sqrt(x)

    def arctan2(self, y, x):
        ctx = x.context
        if isinstance(x, ctx.mpf) and isinstance(y, ctx.mpf):
            if ctx.isinf(x) and ctx.isinf(y):
                # Workaround mpmath 1.3 bug in atan2(+-inf, +-inf)
                # evaluation (see mpmath/mpmath#825)
                pi = ctx.pi
                if x > 0:
                    return pi / 4 if y > 0 else -pi / 4
                return 3 * pi / 4 if y > 0 else -3 * pi / 4
            return ctx.atan2(y, x)
        return ctx.nan

    def angle(self, x):
        ctx = x.context
        if isinstance(x, ctx.mpc):
            return self.arctan2(x.imag, x.real)
        return ctx.nan

    def arctanh(self, x):
        ctx = x.context
        if isinstance(x, ctx.mpc):
            pi = ctx.pi
            zero = ctx.zero
            if ctx.isinf(x.real) or ctx.isinf(x.imag):
                if x.imag < 0:
                    return ctx.make_mpc((zero._mpf_, (-pi / 2)._mpf_))
                return ctx.make_mpc((zero._mpf_, (pi / 2)._mpf_))
        r = ctx.atanh(x)
        if isinstance(x, ctx.mpc):
            # The following if-block ensures compatibiliy with
            # numpy.arctanh.
            if x.imag == 0 and x.real > 1:
                return r.conjugate()
        if isinstance(x, ctx.mpf) and isinstance(r, ctx.mpc):
            # otherwise, mpmath.atanh would return complex value
            return ctx.nan
        return r

    def arctan(self, x):
        ctx = x.context
        if isinstance(x, ctx.mpc):
            pi = ctx.pi
            zero = ctx.zero
            if ctx.isinf(x.real) or ctx.isinf(x.imag):
                if x.real < 0:
                    return ctx.make_mpc(((-pi / 2)._mpf_, zero._mpf_))
                return ctx.make_mpc(((pi / 2)._mpf_, zero._mpf_))
        r = ctx.atan(x)
        if isinstance(x, ctx.mpc):
            # The following if-block ensures compatibiliy with
            # numpy.arctan:
            if x.real == 0 and x.imag < -1:
                return ctx.make_mpc(((-r.real)._mpf_, r.imag._mpf_))
        if isinstance(x, ctx.mpf) and isinstance(r, ctx.mpc):
            # otherwise, mpmath.atan would return complex value
            return ctx.nan
        return r


class numpy_with_mpmath:
    """Namespace of universal functions on numpy arrays that use mpmath
    backend for evaluation and return numpy arrays as outputs.
    """

    _vfunc_cache = dict()

    def __init__(self, **params):
        self.params = params

    def __getattr__(self, name):
        name = dict(asinh="arcsinh", acos="arccos", asin="arcsin", acosh="arccosh", atan="arctan", atanh="arctanh").get(
            name, name
        )
        key = name, tuple(sorted(self.params.items()))
        if key in self._vfunc_cache:
            return self._vfunc_cache[key]
        if hasattr(mpmath_array_api, name):
            vfunc = vectorize_with_mpmath(getattr(mpmath_array_api(), name), **self.params)
            self._vfunc_cache[key] = vfunc
            return vfunc
        raise NotImplementedError(f"vectorize_with_mpmath.{name}")

    def normalize(self, exact, reference, value):
        """Normalize reference and value using precision defined by the
        difference of exact and reference.
        """

        def worker(ctx, s, e, r, v):
            ss, sm, se, sbc = s._mpf_
            es, em, ee, ebc = e._mpf_
            rs, rm, re, rbc = r._mpf_
            vs, vm, ve, vbc = v._mpf_

            if not (ctx.isfinite(e) and ctx.isfinite(r) and ctx.isfinite(v)):
                return r, v

            me = min(se, ee, re, ve)

            # transform mantissa parts to the same exponent base
            sm_e = sm << (se - me)
            em_e = em << (ee - me)
            rm_e = rm << (re - me)
            vm_e = vm << (ve - me)

            # find matching higher and non-matching lower bits of e and r
            sm_b = bin(sm_e)[2:] if sm_e else ""
            em_b = bin(em_e)[2:] if em_e else ""
            rm_b = bin(rm_e)[2:] if rm_e else ""
            vm_b = bin(vm_e)[2:] if vm_e else ""

            m = max(len(sm_b), len(em_b), len(rm_b), len(vm_b))
            em_b = "0" * (m - len(em_b)) + em_b
            rm_b = "0" * (m - len(rm_b)) + rm_b

            c1 = 0
            for b0, b1 in zip(em_b, rm_b):
                if b0 != b1:
                    break
                c1 += 1
            c0 = m - c1

            # truncate r and v mantissa
            rm_m = rm_e >> c0
            vm_m = vm_e >> c0

            # normalized r and v
            nr = ctx.make_mpf((rs, rm_m, -c1, len(bin(rm_m)) - 2)) if rm_m else (-ctx.zero if rs else ctx.zero)
            nv = ctx.make_mpf((vs, vm_m, -c1, len(bin(vm_m)) - 2)) if vm_m else (-ctx.zero if vs else ctx.zero)

            return nr, nv

        ctx = exact.context
        scale = abs(exact)
        if isinstance(exact, ctx.mpc):
            rr, rv = worker(ctx, scale, exact.real, reference.real, value.real)
            ir, iv = worker(ctx, scale, exact.imag, reference.imag, value.imag)
            return ctx.make_mpc((rr._mpf_, ir._mpf_)), ctx.make_mpc((rv._mpf_, iv._mpf_))
        elif isinstance(exact, ctx.mpf):
            return worker(ctx, scale, exact, reference, value)
        else:
            assert 0  # unreachable


class numpy_with_jax:
    """Namespace of universal functions on numpy arrays that use jax
    backend for evaluation and return numpy arrays as outputs.
    """

    _vfunc_cache = dict()

    def __init__(self, **params):
        self.params = params

    def __getattr__(self, name):
        name = dict(asinh="arcsinh", acos="arccos", asin="arcsin", acosh="arccosh", atan="arctan", atanh="arctanh").get(
            name, name
        )
        key = name, tuple(sorted(self.params.items()))
        if key in self._vfunc_cache:
            return self._vfunc_cache[key]
        import jax

        vfunc = vectorize_with_jax(getattr(jax.numpy, name), **self.params)
        self._vfunc_cache[key] = vfunc
        return vfunc


class numpy_with_numpy:
    """Namespace of universal functions on numpy arrays that use numpy
    backend for evaluation and return numpy arrays as outputs.
    """

    _vfunc_cache = dict()

    def __init__(self, **params):
        self.params = params

    def __getattr__(self, name):
        name = dict(asinh="arcsinh", acos="arccos", asin="arcsin", acosh="arccosh", atan="arctan", atanh="arctanh").get(
            name, name
        )
        key = name, tuple(sorted(self.params.items()))
        if key in self._vfunc_cache:
            return self._vfunc_cache[key]

        vfunc = numpy.vectorize(getattr(numpy, name), **self.params)
        self._vfunc_cache[key] = vfunc
        return vfunc


class numpy_with_algorithms:
    """Namespace of universal functions on numpy arrays that use
    functional_algorithms definitions of algorithms.
    """

    _vfunc_cache = dict()

    def __init__(self, **params):
        self.params = params

    def __getattr__(self, name):
        name = dict(arcsinh="asinh", arccos="acos", arcsin="asin", arccosh="acosh", arctan="atan", arctanh="atanh").get(
            name, name
        )
        dtype = self.params["dtype"]
        key = name, tuple(sorted(self.params.items()))
        if key in self._vfunc_cache:
            return self._vfunc_cache[key]

        import functional_algorithms as fa

        ctx = fa.Context(paths=[fa.algorithms])
        graph = ctx.trace(getattr(fa.algorithms, name), dtype)
        graph2 = graph.rewrite(fa.targets.numpy, fa.rewrite)
        func = fa.targets.numpy.as_function(graph2, debug=self.params.get("debug", 0))
        vfunc = vectorize_with_numpy(func)
        self._vfunc_cache[key] = vfunc
        return vfunc


def extra_samples(name, dtype):
    """Return a list of samples that are special to a given function.

    Parameters
    ----------
    name: str
      The name of a function
    dtype:
      Floating-point or complex dtype

    Returns
    -------
    values: list
      Values of function inputs.
    """
    is_complex = "complex" in str(dtype)
    is_float = "float" in str(dtype)
    assert is_float or is_complex, dtype
    values = []
    # Notice that real/complex_samples already include special values
    # such as 0, -inf, inf, smallest subnormals or normals, so don't
    # specify these here.
    if is_float:
        if name in {"acos", "asin"}:
            for v in [-1, 1]:
                values.append(numpy.nextafter(v, v - 1, dtype=dtype))
                values.append(v)
                values.append(numpy.nextafter(v, v + 1, dtype=dtype))
        if name == "log1p":
            for v in [-1, 1]:
                values.append(numpy.nextafter(v, v - 1, dtype=dtype))
                values.append(v)
                values.append(numpy.nextafter(v, v + 1, dtype=dtype))
    if is_complex:
        fdtype = {numpy.complex64: numpy.float32, numpy.complex128: numpy.float64}[dtype]
        if name == "absolute":
            values.append(1.0011048e35 + 3.4028235e38j)
        if name == "log1p":
            # samples z=x+I*y satisfying
            #   x = -0.5 * y**2
            #   abs(y) < 1
            # cause catastrophic cancellation errors
            # when evaluating the real part of complex log1p(z):
            #   0.5*log1p(2*x + x**2 + y**2)
            #
            # The following samples are close to these problematic
            # samples:
            mnexp, mxexp = -8, 2
            for y in numpy.logspace(mnexp, mxexp, num=(mxexp - mnexp + 1), dtype=fdtype):
                x = -fdtype(0.5) * y * y
                values.append(complex(x, y))
                values.append(complex(numpy.nextafter(x, fdtype(-1)), y))
                values.append(complex(numpy.nextafter(x, fdtype(1)), y))

            values.append(-0.24246988 - 0.49096385j)
        if name == "atanh":
            values.extend(
                [
                    -0.9939759 - 0.00042062782j,
                    0.9939759 - 0.00085302145j,
                    -0.73795176 + 0.0877259j,
                    315387600000000 + 36880000000000j,
                ]
            )
    return numpy.array(values, dtype=dtype)


def real_samples(
    size=10,
    dtype=numpy.float32,
    include_infinity=True,
    include_zero=True,
    include_subnormal=False,
    include_nan=False,
    include_huge=True,
    nonnegative=False,
    min_value=None,
    max_value=None,
    unique=True,
):
    """Return a 1-D array of real line samples.

    Parameters
    ----------
    size : int
      Initial size of the samples array. A minimum value is 6. The
      actual size of the returned array may differ from size, except
      when both min/max_value are specified with the same sign.
    dtype:
      Floating-point type: float32, float64.
    include_infinity: bool
      When True, samples include signed infinities.
    include_zero: bool
      When True, samples include zero.
    include_subnormal: bool
      When True, samples include subnormal numbers.
    include_nan: bool
      When True, samples include nan.
    include_huge: bool
      When True, samples include a value that has 1 ULP smaller than
      maximal value.
    nonnegative: bool
      When True, finite samples are all non-negative.
    min_value, max_value: dtype, None
      When min_value or max_value is specified, use for constructing
      uniform samples. Parameters include_infinity, include_nan,
      include_huge, and nonnegative are silently ignored.
    unique: bool
      When True, all samples are unique. Otherwise, allow repeated
      sample values for predictability of the number of samples.
    """
    if isinstance(dtype, str):
        dtype = getattr(numpy, dtype)
    assert dtype in {numpy.float16, numpy.float32, numpy.float64}, dtype
    utype = {numpy.float16: numpy.uint16, numpy.float32: numpy.uint32, numpy.float64: numpy.uint64}[dtype]
    fi = numpy.finfo(dtype)
    size = min(size, 2 ** {numpy.float16: 16, numpy.float32: 64, numpy.float64: 64}[dtype])
    user_specified_bounds = min_value is not None or max_value is not None
    num = size // 2 if not nonnegative and not user_specified_bounds else size
    if include_infinity and not user_specified_bounds:
        num -= 1

    min_pos_value = dtype(fi.smallest_subnormal if include_subnormal else fi.smallest_normal)
    if min_value is None:
        min_value = min_pos_value
    else:
        min_value = dtype(min_value)

    if max_value is None:
        max_value = dtype(fi.max)
    else:
        max_value = dtype(max_value)

    if not include_subnormal:
        if min_value != 0 and abs(min_value) < min_pos_value:
            if min_value < 0:
                min_value = -dtype(min_pos_value)
            else:
                min_value = dtype(0)

        if max_value != 0 and abs(max_value) < min_pos_value:
            if max_value < 0:
                max_value = -dtype(0)
            else:
                max_value = dtype(min_pos_value)

    if min_value == max_value:
        return numpy.array([min_value], dtype=dtype)

    if min_value >= max_value:
        raise ValueError(f"minimal value (={min_value}) cannot be greater than maximal value (={max_value})")

    if user_specified_bounds:
        if min_value >= dtype(0):
            start, end = min_value.view(utype), max_value.view(utype)
            step = int(end - start)
            r = (start + numpy.array([i // (num - 1) for i in range(0, num * step, step)], dtype=utype)).view(dtype)
            assert r.size == num
        elif max_value <= -dtype(0):
            start, end = max_value.view(utype), min_value.view(utype)
            step = int(end - start)
            r = (start + numpy.array([i // (num - 1) for i in range(0, num * step, step)], dtype=utype)).view(dtype)
            assert r.size == num
            r = r[::-1]
        else:
            neg_diff = diff_ulp(abs(min_value), min_pos_value)
            pos_diff = diff_ulp(abs(max_value), min_pos_value)
            neg_num = int(neg_diff * num / max(1, neg_diff + pos_diff))
            pos_num = num - neg_num - int(bool(include_zero))

            neg_part = real_samples(
                size=neg_num,
                dtype=dtype,
                include_subnormal=include_subnormal,
                min_value=min_value,
                max_value=-min_pos_value if min_value < -min_pos_value else -dtype(0),
            )
            pos_part = real_samples(
                size=pos_num,
                dtype=dtype,
                include_subnormal=include_subnormal,
                min_value=min_pos_value if min_pos_value < max_value else dtype(0),
                max_value=max_value,
            )
            if include_zero:
                r = numpy.concatenate([neg_part, numpy.array([0], dtype=dtype), pos_part])
            else:
                r = numpy.concatenate([neg_part, pos_part])
        if not include_subnormal:
            for i in range(len(r)):
                if r[i] == 0:
                    continue
                if abs(r[i]) < fi.smallest_normal:
                    if r[i] < 0:
                        r[i] = -dtype(0)
                    else:
                        r[i] = dtype(0)
        return numpy.unique(r) if unique else r
    if 1:
        # The following method gives a sample distibution that is
        # uniform with respect to ULP distance between positive
        # neighboring samples
        start, end = min_value.view(utype), max_value.view(utype)
        step = int(end - start)
        finite_positive = (start + numpy.array([i // (num - 1) for i in range(0, num * step, step)], dtype=utype)).view(dtype)
    else:
        start = fi.minexp + fi.negep + 1 if include_subnormal else fi.minexp
        end = fi.maxexp
        with warnings.catch_warnings(action="ignore"):
            # Note that logspace gives samples distribution that is
            # approximately uniform with respect to ULP distance between
            # neighboring normal samples. For subnormal samples, logspace
            # produces repeated samples that will be eliminated below via
            # numpy.unique.
            finite_positive = numpy.logspace(start, end, base=2, num=num, dtype=dtype)
    finite_positive[-1] = max_value

    if include_huge and num > 3:
        huge = -numpy.nextafter(-max_value, numpy.inf, dtype=dtype)
        finite_positive[-2] = huge

    parts = []
    extra = []
    if not nonnegative:
        if include_infinity:
            extra.append(-numpy.inf)
        parts.append(-finite_positive[::-1])
    if include_zero:
        extra.append(0)
    parts.append(finite_positive)
    if include_infinity:
        extra.append(numpy.inf)
    if include_nan:
        extra.append(numpy.nan)
    parts.append(numpy.array(extra, dtype=dtype))

    # Using unique because logspace produces repeated subnormals when
    # size is large
    r = numpy.concatenate(parts)
    return numpy.unique(r) if unique else r


def periodic_samples(
    period=1,
    size=10,
    dtype=numpy.float32,
    periods=5,
    include_subnormal=False,
    unique=False,
):
    """Return a 1-D array of real line samples that contains intervals of
    length period distributed ULP-uniformly over the real line.
    """
    eps = dtype(numpy.finfo(dtype).epsneg)
    # The number of floating-point values between x - period and x is
    #   log2(x / (x-period)) / eps
    # that must match with size for the largest sample x:
    #   log2(max_value / (max_value - period)) == size * eps
    # Hence:
    max_value = dtype(period / (1 - 2 ** -(eps * size)))
    left_points = []
    left_points.extend(
        real_samples(size=periods // 2, min_value=-max_value, max_value=-period - period, dtype=dtype, unique=True)
    )
    if periods % 2:
        left_points.append(-period / dtype(2))
    left_points.extend(
        real_samples(size=periods // 2, min_value=period, max_value=max_value - period, dtype=dtype, unique=True)
    )
    return numpy.concatenate(
        tuple(
            real_samples(
                size=size,
                min_value=left_point,
                max_value=left_point + period,
                dtype=dtype,
                include_subnormal=include_subnormal,
                unique=unique,
            )
            for left_point in left_points
        )
    )


def complex_samples(
    size=(10, 10),
    dtype=numpy.float32,
    include_infinity=True,
    include_zero=True,
    include_subnormal=False,
    include_nan=False,
    include_huge=True,
    nonnegative=False,
    min_real_value=None,
    max_real_value=None,
    min_imag_value=None,
    max_imag_value=None,
):
    """Return a 2-D array of complex plane samples.

    Parameters
    ----------
    size : tuple(int, int)
      Initial size of the samples array. A minimum value is (6, 6).
      The actual size of the returned array may differ from size.
    dtype:
    include_infinity: bool
    include_zero: bool
    include_subnormal: bool
    include_nan: bool
    nonnegative: bool
    min_real_value, max_real_value, min_imag_value, max_imag_value: dtype
    """
    if isinstance(dtype, str):
        dtype = getattr(numpy, dtype)
    dtype = {numpy.complex64: numpy.float32, numpy.complex128: numpy.float64}.get(dtype, dtype)
    re = real_samples(
        size[0],
        dtype=dtype,
        include_infinity=include_infinity,
        include_zero=include_zero,
        include_subnormal=include_subnormal,
        include_nan=include_nan,
        include_huge=include_huge,
        nonnegative=nonnegative,
        min_value=min_real_value,
        max_value=max_real_value,
    )
    im = real_samples(
        size[1],
        dtype=dtype,
        include_infinity=include_infinity,
        include_zero=include_zero,
        include_subnormal=include_subnormal,
        include_nan=include_nan,
        include_huge=include_huge,
        nonnegative=nonnegative,
        min_value=min_imag_value,
        max_value=max_imag_value,
    )
    complex_dtype = {numpy.float32: numpy.complex64, numpy.float64: numpy.complex128}[dtype]
    real_part = re.reshape((-1, re.size)).repeat(im.size, 0).astype(complex_dtype)
    imag_part = im.repeat(2).view(complex_dtype)
    imag_part.real[:] = 0
    imag_part = imag_part.reshape((im.size, -1)).repeat(re.size, 1)
    return real_part + imag_part  # TODO: avoid arithmetic operations


def real_pair_samples(
    size=(10, 10),
    dtype=numpy.float32,
    include_infinity=True,
    include_zero=True,
    include_subnormal=False,
    include_nan=False,
    include_huge=True,
    nonnegative=False,
):
    """Return a pair of 1-D arrays of real line samples.

    Parameters
    ----------
    size : tuple(int, int)
      Initial size of the samples array. A minimum value is (6, 6). The
      actual size of the returned array is approximately size[0] * size[1].
    dtype:
    include_infinity: bool
    include_zero: bool
    include_subnormal: bool
    include_nan: bool
    nonnegative: bool
    """
    s1 = real_samples(
        size=size[0],
        dtype=dtype,
        include_infinity=include_infinity,
        include_zero=include_zero,
        include_subnormal=include_subnormal,
        include_nan=include_nan,
        nonnegative=nonnegative,
        include_huge=include_huge,
    )
    s2 = real_samples(
        size=size[1],
        dtype=dtype,
        include_infinity=include_infinity,
        include_zero=include_zero,
        include_subnormal=include_subnormal,
        include_nan=include_nan,
        nonnegative=nonnegative,
        include_huge=include_huge,
    )
    s1, s2 = s1.reshape(1, -1).repeat(s2.size, 0).flatten(), s2.repeat(s1.size)
    return s1, s2


def complex_pair_samples(
    size=((10, 10), (10, 10)),
    dtype=numpy.float32,
    include_infinity=True,
    include_zero=True,
    include_subnormal=False,
    include_nan=False,
    include_huge=True,
    nonnegative=False,
):
    """Return a pair of 2-D arrays of complex plane samples.

    Parameters
    ----------
    size : tuple(tuple(int, int), tuple(int, int))
      Initial size of the samples array. A minimum value is ((6, 6),
      (6, 6)). The actual size of the returned array is approximately
      (size[0][0] * size[1][0], size[0][1] * size[1][1]).
    dtype:
    include_infinity: bool
    include_zero: bool
    include_subnormal: bool
    include_nan: bool
    nonnegative: bool
    """
    s1 = complex_samples(
        size=size[0],
        dtype=dtype,
        include_infinity=include_infinity,
        include_zero=include_zero,
        include_subnormal=include_subnormal,
        include_nan=include_nan,
        include_huge=include_huge,
        nonnegative=nonnegative,
    )
    s2 = complex_samples(
        size=size[1],
        dtype=dtype,
        include_infinity=include_infinity,
        include_zero=include_zero,
        include_subnormal=include_subnormal,
        include_nan=include_nan,
        include_huge=include_huge,
        nonnegative=nonnegative,
    )
    shape1 = s1.shape
    shape2 = s2.shape
    s1, s2 = numpy.tile(s1, shape2), s2.repeat(shape1[0], 0).repeat(shape1[1], 1)
    return s1, s2


def expansion_samples(
    size=None,
    length=2,
    overlapping=False,
    dtype=numpy.float32,
    min_value=None,
    max_value=None,
    include_subnormal=False,
    nonnegative=False,
):
    """Return a list of FP expansion samples.

    A FP expansion is a list of floating-point values with specified
    overlapping property.

    Parameters
    ----------
    size : int
      Initial size of the samples array. A minimum value is 6 ** length.
    length : int
      The length of an FP expansion.
    dtype:
      Floating-point type: float16, float32, float64.
    include_subnormal: bool
      When True, samples include subnormal numbers.
    nonnegative: bool
      When True, finite samples are all non-negative.

    """
    if size is None:
        size = 6**length
    assert length > 0
    if length == 1:
        return [
            [x_]
            for x_ in real_samples(
                size,
                dtype=dtype,
                include_infinity=False,
                include_huge=False,
                include_subnormal=include_subnormal,
                nonnegative=nonnegative,
                min_value=min_value,
                max_value=max_value,
            )
        ]

    size1 = max(6, 2 ** int(math.log2(size) / length))
    size2 = max(6 ** (length - 1), int(size // size1 + 1))
    fi = numpy.finfo(dtype)
    c1 = numpy.ldexp(dtype(1), fi.negep + 1)
    c6 = numpy.ldexp(dtype(1), fi.negep + 6)
    c = numpy.ldexp(dtype(1), fi.negep)
    lst = []
    for x in expansion_samples(
        size=size1,
        length=1,
        overlapping=overlapping,
        dtype=dtype,
        include_subnormal=include_subnormal,
        nonnegative=nonnegative,
        min_value=min_value,
        max_value=max_value,
    ):
        if overlapping:
            inner_lst = expansion_samples(
                size=size2 // 2 + 1,
                length=length - 1,
                overlapping=overlapping,
                dtype=dtype,
                include_subnormal=include_subnormal,
                nonnegative=nonnegative,
                min_value=-abs(x[0]) * c6,
                max_value=-abs(x[0]) * c1,
            ) + expansion_samples(
                size=size2 // 2 + 1,
                length=length - 1,
                overlapping=overlapping,
                dtype=dtype,
                include_subnormal=include_subnormal,
                nonnegative=nonnegative,
                min_value=abs(x[0]) * c1,
                max_value=abs(x[0]) * c6,
            )
        else:
            inner_lst = expansion_samples(
                size=size2,
                length=length - 1,
                overlapping=overlapping,
                dtype=dtype,
                include_subnormal=include_subnormal,
                nonnegative=nonnegative,
                min_value=-abs(x[0]) * c,
                max_value=abs(x[0]) * c,
            )
        for y in inner_lst:
            e = x + y
            with warnings.catch_warnings(action="ignore"):
                s = sum(e[:1], e[-1])
            if numpy.isfinite(s):
                lst.append(e)
    return lst


def iscomplex(value):
    return isinstance(value, (complex, numpy.complexfloating))


def isfloat(value):
    return isinstance(value, (float, numpy.floating))


def diff_log2ulp(x, y, flush_subnormals=UNSPECIFIED, equal_nan=False) -> int:
    """Return log2-ULP distance between two floating point numbers:

    diff_log2ulp(x, y) = diff_ulp(x, y).bit_length()
    """
    return diff_ulp(x, y, flush_subnormals=flush_subnormals, equal_nan=equal_nan).bit_length()


def diff_ulp(x, y, flush_subnormals=UNSPECIFIED, equal_nan=False) -> int:
    """Return ULP distance between two floating point numbers.

    For complex inputs, return largest ULP among real and imaginary
    parts.

    When flush_subnormals is set to True, ULP difference does not
    account for subnormals while subnormal values are rounded to
    nearest normal, ties to even.

    When equal_nan is set to True, ULP difference between nan values
    of both quiet and signaling kinds is defined as 0.
    """
    if isinstance(x, list) and isinstance(y, list):
        if len(x) < len(y):
            x = x + [type(x[0])(0)] * (len(y) - len(x))
        elif len(x) > len(y):
            y = y + [type(y[0])(0)] * (len(x) - len(y))
        u = 0
        for x_, y_ in zip(x, y):
            u += diff_ulp(x_, y_, flush_subnormals=flush_subnormals, equal_nan=equal_nan)
        return u
    if isinstance(x, numpy.floating):
        uint = {numpy.float64: numpy.uint64, numpy.float32: numpy.uint32, numpy.float16: numpy.uint16}[x.dtype.type]
        sx = -1 if x < 0 else (1 if x > 0 else 0)
        sy = -1 if y < 0 else (1 if y > 0 else 0)
        x, y = abs(x), abs(y)
        ix, iy = int(x.view(uint)), int(y.view(uint))
        if numpy.isfinite(x) and numpy.isfinite(y):
            flush_subnormals = flush_subnormals if flush_subnormals is not UNSPECIFIED else default_flush_subnormals
            if flush_subnormals:
                fi = numpy.finfo(x.dtype)
                i = int(fi.smallest_normal.view(uint)) - 1  # 0 distance to largest subnormal
                ix = ix - i if ix > i else (0 if 2 * ix <= i else 1)
                iy = iy - i if iy > i else (0 if 2 * iy <= i else 1)
            if sx != sy:
                # distance is measured through 0 value
                result = ix + iy
            else:
                result = ix - iy if ix >= iy else iy - ix
            return result
        elif ix == iy and sx == sy:
            return 0
        elif numpy.isnan(x) and numpy.isnan(y):
            if equal_nan:
                return 0
        return {numpy.float64: 2**64, numpy.float32: 2**32, numpy.float16: 2**16}[x.dtype.type]
    elif isinstance(x, numpy.complexfloating):
        return max(
            diff_ulp(x.real, y.real, flush_subnormals=flush_subnormals, equal_nan=equal_nan),
            diff_ulp(x.imag, y.imag, flush_subnormals=flush_subnormals, equal_nan=equal_nan),
        )
    elif isinstance(x, numpy.ndarray) and isinstance(y, numpy.ndarray):
        if x.shape == () and y.shape == ():
            return numpy.array(diff_ulp(x[()], y[()], flush_subnormals=flush_subnormals, equal_nan=equal_nan))
        assert x.shape == y.shape, (x.shape, y.shape)
        return numpy.array([diff_ulp(x_, y_, flush_subnormals=flush_subnormals, equal_nan=equal_nan) for x_, y_ in zip(x, y)])
    elif isinstance(x, numpy.ndarray) and isinstance(y, (numpy.complexfloating, numpy.floating)):
        if x.shape == ():
            return numpy.array(diff_ulp(x[()], y, flush_subnormals=flush_subnormals, equal_nan=equal_nan))
    elif isinstance(x, (numpy.complexfloating, numpy.floating)) and isinstance(y, numpy.ndarray):
        if y.shape == ():
            return numpy.array(diff_ulp(x, y[()], flush_subnormals=flush_subnormals, equal_nan=equal_nan))
    raise NotImplementedError((type(x), type(y)))


def make_complex(r, i):
    if r.dtype == numpy.float32 and i.dtype == numpy.float32:
        return numpy.array([r, i]).view(numpy.complex64)[0]
    elif i.dtype == numpy.float64 and i.dtype == numpy.float64:
        return numpy.array([r, i]).view(numpy.complex128)[0]
    raise NotImplementedError((r.dtype, i.dtype))


def mul(x, y):
    # safe multiplication of complex and float numbers where complex
    # number may have non-finite parts
    if iscomplex(x) and isfloat(y):
        return make_complex(mul(x.real, y), mul(x.imag, y))
    elif iscomplex(y) and isfloat(x):
        return make_complex(mul(x, y.real), mul(x, y.imag))
    with warnings.catch_warnings(action="ignore"):
        return numpy.multiply(x, y, dtype=x.dtype)


def left_right_sample_functions(sample):
    if isinstance(sample, numpy.floating):
        inf = type(sample)(numpy.inf)

        def left(sample):
            if numpy.isneginf(sample):
                return sample
            with warnings.catch_warnings(action="ignore"):
                return numpy.nextafter(sample, -inf)

        def right(sample):
            if numpy.isposinf(sample):
                return sample
            with warnings.catch_warnings(action="ignore"):
                return numpy.nextafter(sample, inf)

    elif isinstance(sample, float):

        def left(sample):
            return math.nextafter(sample, -math.inf)

        def right(sample):
            return math.nextafter(sample, math.inf)

    elif isinstance(sample, numpy.ndarray) and issubclass(sample.dtype.type, numpy.floating):
        inf = sample.dtype.type(numpy.inf)

        def left(sample):
            with warnings.catch_warnings(action="ignore"):
                return numpy.nextafter(sample, -inf)

        def right(sample):
            with warnings.catch_warnings(action="ignore"):
                return numpy.nextafter(sample, inf)

    else:
        raise TypeError(f"not implemented for {type(sample)}")

    return left, right


def sample_surrounding(sample, ulps=1):
    """Sample iterator of sample surroundings."""
    if isinstance(sample, numpy.complexfloating):
        for i, re in enumerate(sample_surrounding(sample.real, ulps=ulps)):
            for j, im in enumerate(sample_surrounding(sample.imag, ulps=ulps)):
                u = abs(i - ulps) + abs(j - ulps)
                if u <= ulps:
                    r = numpy.empty((1,), dtype=type(sample))
                    r.real = re
                    r.imag = im
                    yield r[0]
    elif isinstance(sample, complex):
        for i, re in enumerate(sample_surrounding(sample.real, ulps=ulps)):
            for j, im in enumerate(sample_surrounding(sample.imag, ulps=ulps)):
                u = abs(i - ulps) + abs(j - ulps)
                if u <= ulps:
                    yield complex(re, im)
    elif isinstance(sample, tuple):
        samples = tuple(list(sample_surrounding(s, ulps=ulps)) for s in sample)
        yield from itertools.product(*samples)
    elif isinstance(sample, numpy.ndarray) and issubclass(sample.dtype.type, numpy.complexfloating):
        for i, re in enumerate(sample_surrounding(sample.real, ulps=ulps)):
            for j, im in enumerate(sample_surrounding(sample.imag, ulps=ulps)):
                u = abs(i - ulps) + abs(j - ulps)
                if u <= ulps:
                    z = re.astype(sample.dtype.type)
                    z.imag = im
                    yield z
    elif isinstance(sample, list):
        yield from zip(*[list(sample_surrounding(s, ulps=ulps)) for s in sample])
    else:
        left, right = left_right_sample_functions(sample)
        samples = [sample]
        left_sample = sample
        right_sample = sample
        for i in range(ulps):
            left_sample = left(left_sample)
            right_sample = right(right_sample)
            samples.insert(0, left_sample)
            samples.append(right_sample)
        yield from samples


def function_bounds(func, samples, ulps=1, enable_progressbar=False, workers=None):
    """Return the bounds of a function on the samples surroundings.

    If
      lower, upper = function_bounds(func, sample, ulps)
    then
      lower <= func(s) <= upper
    for all s such that
      diff_ulp(sample, s) <= ulps

    For complex values, `a <= b` is defined as `a.real <= b.real and
    a.imag <= b.imag`.
    """
    lower, upper, is_complex = None, None, None
    for samples_ in sample_surrounding(samples, ulps=ulps):
        result = func.call(samples_, workers=workers, enable_progressbar=enable_progressbar)
        if lower is None:
            lower = result.copy()
            upper = result.copy()
            is_complex = issubclass(result.dtype.type, numpy.complexfloating)
        elif is_complex:
            lower.real = numpy.minimum(lower.real, result.real)
            upper.real = numpy.maximum(upper.real, result.real)
            lower.imag = numpy.minimum(lower.imag, result.imag)
            upper.imag = numpy.maximum(upper.imag, result.imag)
        else:
            lower = numpy.minimum(lower, result)
            upper = numpy.maximum(upper, result)
    return lower, upper


def is_within_bounds(value, lower, upper):
    if isinstance(value, (complex, numpy.complexfloating)):
        return is_within_bounds(value.real, lower.real, upper.real) and is_within_bounds(value.imag, lower.imag, upper.imag)
    return lower <= value and value <= upper


def validate_function(
    func,
    reference,
    samples,
    dtype,
    verbose=True,
    flush_subnormals=UNSPECIFIED,
    enable_progressbar=False,
    workers=None,
    max_valid_ulp_count=3,
    max_bound_ulp_width=1,
    bucket=None,
):
    ulp_stats = defaultdict(int)
    outrange_stats = defaultdict(int)

    reference_results = reference.call(samples, workers=workers, enable_progressbar=enable_progressbar)
    if max_bound_ulp_width:
        lower, upper = function_bounds(
            reference, samples, ulps=max_bound_ulp_width, workers=workers, enable_progressbar=enable_progressbar
        )

    for index in tqdm(range(len(samples))) if enable_progressbar else range(len(samples)):
        v2 = reference_results[index][()]
        sample = samples[index]
        v1 = func(*sample) if isinstance(sample, tuple) else func(sample)
        assert v1.dtype == v2.dtype, (sample, v1, v2)
        ulp = diff_ulp(v1, v2, flush_subnormals=flush_subnormals)
        if isinstance(ulp, numpy.ndarray):
            assert ulp.shape == ()
            ulp = ulp[()]
        ulp_stats[ulp] += 1

        if bucket is not None:
            # collect samples and values to a provide bucket
            bucket.add(sample, ulp, v1, v2)

        if max_bound_ulp_width:
            within_bounds = is_within_bounds(v1, lower[index], upper[index])
            if not within_bounds:
                if ulp > max_valid_ulp_count:
                    ulp_stats[-1] += 1
                outrange_stats[ulp] += 1
            if ulp > max_valid_ulp_count and not within_bounds and verbose:
                print(f"--> {sample, v1, v2, ulp, within_bounds, lower[index], upper[index]=}")
        else:
            if ulp > max_valid_ulp_count:
                ulp_stats[-1] += 1
            if ulp > max_valid_ulp_count and verbose:
                print(f"--> {sample, v1, v2, ulp=}")
    return ulp_stats[-1] == 0, dict(ulp=ulp_stats, outrange=outrange_stats)


def function_validation_parameters(func_name, dtype, device=None):
    # If a function has symmetries, exclude superfluous samples by
    # specifying a region of the function domain:
    samples_limits = dict(
        min_real_value=-numpy.inf,
        max_real_value=numpy.inf,
        min_imag_value=-numpy.inf,
        max_imag_value=numpy.inf,
    )

    # diff_ulp(func(sample), reference(sample)) <= max_valid_ulp_count
    max_valid_ulp_count = 3

    # func(sample) is within interval [lower, upper] where
    #
    #   lower = min(reference(s) for s in surrounding(sample) if diff_ulp(s, sample) <= max_bound_ulp_width)
    #   upper = max(reference(s) for s in surrounding(sample) if diff_ulp(s, sample) <= max_bound_ulp_width)
    #
    # By default, we'll skip out-of-ulp-range tests to speed-up function validation:
    max_bound_ulp_width = 0

    # mpmath does not guarantee the accuracy of function evaluation to
    # the given precision. If there are mismatches between functions
    # defined in algorithms.py and mpmath, we'll increase the
    # precision of mpmath to improve the accurancy of reference
    # values:
    extra_prec_multiplier = 1
    if func_name in {"acos", "asin", "asinh", "acosh", "atan", "atanh"}:
        extra_prec_multiplier = 20
    elif func_name == "sqrt":
        extra_prec_multiplier = 2
        if device == "cuda":
            max_valid_ulp_count = 5
        else:
            max_valid_ulp_count = 4
    elif func_name == "log1p":
        extra_prec_multiplier = 10  # remove when mpmath#803 becomes available
        max_valid_ulp_count = 4
    elif func_name in {"log", "log10", "log2"}:
        max_valid_ulp_count = 3
    elif func_name == "exp":
        max_valid_ulp_count = 3
        extra_prec_multiplier = 20
    elif func_name == "expm1":
        max_valid_ulp_count = 3
        extra_prec_multiplier = 20
    elif func_name in {"atanh", "atan"}:
        extra_prec_multiplier = 20
    elif func_name in {"tanh", "tan"}:
        extra_prec_multiplier = 20
    return dict(
        extra_prec_multiplier=extra_prec_multiplier,
        max_valid_ulp_count=max_valid_ulp_count,
        max_bound_ulp_width=max_bound_ulp_width,
        samples_limits=samples_limits,
    )


def format_python(code):
    try:
        import black
    except ImportError:
        return code
    return black.format_str(code, mode=black.FileMode(line_length=127))


def format_cpp(code):
    import pathlib
    import tempfile
    import subprocess

    with tempfile.NamedTemporaryFile(mode="w", suffix=".cc", delete=False) as fp:
        fp.write(code)
        fp.close()

        command = ["clang-format", "--style=Google", fp.name]
        try:
            p = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=None,
                stdin=subprocess.PIPE,
                universal_newlines=True,
            )
        except OSError as e:
            # Give the user more context when clang-format isn't
            # found/isn't executable, etc.
            print('Failed to run "%s" - %s"' % (" ".join(command), e.strerror))
            pathlib.Path(fp.name).unlink(missing_ok=True)
            return code

        stdout, stderr = p.communicate()
        if p.returncode == 0:
            code = stdout

        pathlib.Path(fp.name).unlink(missing_ok=True)

    return code


_warn_once_cache = set()


def warn_once(msg, stacklevel=1):
    if msg in _warn_once_cache:
        return
    _warn_once_cache.add(msg)
    warnings.warn(msg, stacklevel=stacklevel + 1)


class Clusters:

    class Cluster:

        def __init__(self):
            self.points = set()

        def contains(self, point):
            for x, y in self.points:
                for dx in {-1, 0, 1}:
                    for dy in {-1, 0, 1}:
                        if point == (x + dx, y + dy):
                            return True
            return False

        def add(self, point):
            self.points.add(point)

        def merge_from(self, other):
            self.points.update(other.points)

        def __repr__(self):
            return f"{type(self).__name__}({self.points})"

        def center(self):
            sx, sy = 0, 0
            for x, y in self.points:
                sx += x
                sy += y
            return (sx / len(self.points), sy / len(self.points))

        def center_point(self):
            cx, cy = self.center()
            lst = []
            for x, y in self.points:
                lst.append((abs(x - cx) + abs(y - cy), (x, y)))
            return sorted(lst)[0][1]

    def __init__(self):
        self.clusters = []

    def __repr__(self):
        return f"{type(self).__name__}({self.clusters})"

    def add(self, point):
        matching_clusters = []
        for index, cluster in enumerate(self.clusters):
            if cluster.contains(point):
                matching_clusters.append(index)

        if len(matching_clusters) == 0:
            self.clusters.append(Clusters.Cluster())
            self.clusters[-1].add(point)
        elif len(matching_clusters) == 1:
            self.clusters[matching_clusters[0]].add(point)
        else:
            cluster = self.clusters[matching_clusters[0]]
            for index in reversed(matching_clusters[1:]):
                cluster.merge_from(self.clusters[index])
                del self.clusters[index]

    def __iter__(self):
        return iter(self.clusters)

    def split(self):
        """Split clusters by discarding the their center points."""
        result = type(self)()
        for index, cluster in enumerate(self.clusters):
            center = cluster.center_point()
            for point in cluster.points:
                if point == center:
                    continue
                result.add(point)
        return result


def expansion2mpf(ctx, e):
    """Transform FP expansion to mpf instance."""
    return sum([float2mpf(ctx, e_) for e_ in reversed(e[:-1])], float2mpf(ctx, e[-1]))


def mpf2expansion(dtype, x, length=None, functional=False, base=None):
    """Transform mpf instance to an FP expansion."""
    if isinstance(x, list):
        return [mpf2expansion(dtype, x_, length=length, functional=functional, base=base) for x_ in x]
    if x.context.isinf(x):
        lst = [mpf2float(dtype, x)]
    else:
        lst = []
        index = 0
        while True:
            if base is not None:
                coeff = base**index
                y = mpf2float(dtype, x * coeff)
            else:
                y = mpf2float(dtype, x)
            lst.append(y)
            if numpy.isinf(y) or y == 0:
                break
            if base is not None:
                x = x - float2mpf(x.context, y) / coeff
            else:
                x = x - float2mpf(x.context, y)
            if length is not None and len(lst) == length:
                break
            index += 1

    if functional and length is not None and len(lst) < length:
        lst.extend([dtype(0)] * (length - len(lst)))

    return lst


def fraction2expansion(dtype, q, length=2, functional=False, base=None):
    """Transform a fraction to an FP expansion.

    If the length of output is smaller than the specified length then
    the conversion is exact.

    The output may require renormalization.
    """
    lst = []
    index = 0
    while True:
        if base is not None:
            coeff = base**index
            f = fraction2float(dtype, q * coeff)
        else:
            f = fraction2float(dtype, q)
        if f == dtype(0):
            break
        lst.append(f)
        if base is not None:
            q = q - float2fraction(f) / coeff
        else:
            q = q - float2fraction(f)
        if length is not None and len(lst) == length:
            break
        index += 1
    if functional and len(lst) < length:
        lst.extend([dtype(0)] * (length - len(lst)))
    return lst


def float2expansion(dtype, q, length=None, functional=False, base=None):
    assert base is None  # not impl
    lst = []
    while True:
        f = dtype(q)
        if f == dtype(0):
            break
        lst.append(f)
        q = q - type(q)(f)
        if length is not None and len(lst) == length:
            break
    if functional and len(lst) < length:
        lst.extend([dtype(0)] * (length - len(lst)))
    return lst


def number2expansion(dtype, q, length=None, functional=False, base=None):
    if isinstance(q, list):
        return [number2expansion(dtype, q_, length=length, functional=functional, base=base) for q_ in q]
    if isinstance(q, int):
        q = fractions.Fraction(q, 1)
    if isinstance(q, fractions.Fraction):
        return fraction2expansion(dtype, q, length=length, functional=functional, base=base)
    if isinstance(q, mpmath.mpf):
        return mpf2expansion(dtype, q, length=length, functional=functional, base=base)
    if isinstance(q, (numpy.floating, float)):
        return float2expansion(dtype, q, length=length, functional=functional, base=base)
    assert 0, type(q)  # not implemented


def multiword2mpf(ctx, mw):
    """Transform multiword to mpf instance."""
    s = float2mpf(ctx, mw[-1])
    for i in reversed(range(len(mw) - 1)):
        s = s + float2mpf(ctx, mw[i])
    return s


def mpf2multiword(dtype, x, p=None, max_length=None):
    """Return a list of fixed-width floating point numbers such that

      x == sum([float2mpf(mpmath.mp, v) for v in result]) + O(smallest subnormal of dtype)

    where x is a mpmath mpf instance.

    In general, the result list is a monotone sequence of
    floating-point numbers with precision p. If p is unspecified then
    p is taken as the precision of the dtype floating-point system.

    The result list is an exact representation of x when

      x.bc <= p * len(result)

    holds. Otherwise, the result list represents a truncated x because
    the given dtype floating-point system has limited exponent
    size. For example, x is truncated when x.bc is greater than

         33 for float16 (maximal non-truncated len(result) is 3),
        168 for float32 (maximal non-truncated len(result) is 7),
       1113 for float64 (maximal non-truncated len(result) is 21),
      16384 for longdouble (maximal non-truncated len(result) is 256),

    respectively.

    When max_length is specified, then tailing words may be
    accumulated so that

      len(result) <= max_length

    holds.
    """
    sign, man, exp, bc = x._mpf_
    mpf = x.context.mpf
    tp = get_precision(dtype)
    if p is None:
        p = tp
    if p > tp:
        raise ValueError(f"specified precision ({p}) exceeds the precision of {dtype.__name__} ({tp})")
    bl = man.bit_length()
    mask = (1 << min(bl, p)) - 1
    result = []
    offset = max(bl - p, 0)
    if max_length is not None and max_length == 1:
        result.append(mpf2float(dtype, x))
        offset = 0
    while True:
        man1 = (man & (mask << offset)) >> offset
        bl1 = man1.bit_length()
        d = mask.bit_length() - bl1
        assert d >= 0
        if d > 0 and offset >= d:
            # skip heading bytes that are zero for optimal compression
            # of bit data. In some cases, this reduces result length.
            offset -= d
            man1 = (man & (mask << offset)) >> offset
            bl1 = man1.bit_length()
        exp1 = exp + offset
        x1 = mpf2float(dtype, mpf((sign, man1, exp1, bl1)))
        if x1 == dtype(0):
            # result represents truncated x
            break
        result.append(x1)
        if offset == 0:
            break
        elif (max_length is not None and len(result) == max_length - 1) or (offset < p):
            # prepare for the last item
            mask = (1 << offset) - 1
            offset = 0
        else:
            # shift to next item
            offset -= bl1

    if max_length is not None:
        assert len(result) <= max_length, (len(result), max_length)

    return result


class FractionContext:
    """A light-weight context for evaluating select with Fraction inputs."""

    def constant(self, value, like):
        if isinstance(like, fractions.Fraction):
            return fractions.Fraction(value)
        assert 0, (value, like, type(like))  # unreachable

    def reciprocal(self, x):
        if isinstance(x, fractions.Fraction):
            return 1 / x


class NumpyContext:
    """A light-weight context for evaluating select with numpy inputs."""

    def __init__(self, default_constant_type=None, mpmath_context=None):
        self._default_like = None
        self._default_constant_type = default_constant_type
        self._mpmath_context = mpmath_context

    @property
    def default_like(self):
        if self._default_like is None:
            if self._default_constant_type is not None:
                self._default_like = self._default_constant_type(0)
            else:
                raise ValueError("default_constant_type not specified")
        return self._default_like

    def _is_nonzero(self, value):
        if isinstance(value, numpy.floating):
            return not (value == 0)
        if isinstance(value, numpy.ndarray):
            return value != 0
        assert 0, (value, type(value))  # unreachable

    def logical_not(self, a):
        if isinstance(a, numpy.ndarray):
            return numpy.logical_not(a)
        return not a

    def logical_or(self, a, b):
        if isinstance(a, numpy.ndarray) or isinstance(b, numpy.ndarray):
            return numpy.logical_or(a, b)
        return a or b

    def logical_xor(self, a, b):
        if isinstance(a, numpy.ndarray) or isinstance(b, numpy.ndarray):
            return numpy.logical_xor(a, b)
        return (a and not b) or (not a and b)

    def logical_and(self, a, b):
        if isinstance(a, numpy.ndarray) or isinstance(b, numpy.ndarray):
            return numpy.logical_and(a, b)
        return a and b

    def isnan(self, value):
        if isinstance(value, (numpy.floating, numpy.ndarray)):
            return numpy.isnan(value)
        assert 0, (value, type(value))  # unreachable

    def select(self, cond, x, y):
        if isinstance(cond, (bool, numpy.bool_)):
            return x if cond else y
        if isinstance(cond, numpy.ndarray):
            return numpy.where(cond, x, y)
        assert 0, (type(cond), type(x), type(y))

    def list(self, lst):
        return list(lst)

    def constant(self, value, like=None):
        if like is None:
            assert isinstance(value, (int, float, numpy.floating, str, numpy.ndarray)), type(value)
            like = self.default_like
        if isinstance(like, numpy.floating):
            dtype = type(like)
            if isinstance(value, dtype):
                return value
            elif isinstance(value, str):
                if value == "inf":
                    return dtype(numpy.inf)
                elif value == "largest":
                    return numpy.finfo(dtype).max
                elif value == "smallest":
                    return numpy.finfo(dtype).smallest_normal
                elif value == "pi":
                    return dtype(numpy.pi)
                assert 0, (value, dtype)  # not implemented
            elif isinstance(value, fractions.Fraction):
                return mpf2float(dtype, fraction2mpf(mpmath, value))
            return dtype(value)
        if isinstance(like, numpy.ndarray):
            return self.constant(value, like=like.dtype.type(0))

        assert 0, (value, like, type(like))  # unreachable

    def ne(self, x, y):
        if isinstance(x, numpy.floating) or isinstance(y, numpy.floating):
            return x != y
        assert 0, (x, y, type(x))  # unreachable

    def eq(self, x, y):
        if isinstance(x, numpy.floating) or isinstance(y, numpy.floating):
            return x == y
        assert 0, (x, y, type(x))  # unreachable

    def lt(self, x, y):
        if isinstance(x, numpy.floating) or isinstance(y, numpy.floating):
            return x < y
        assert 0, (x, y, type(x))  # unreachable

    def floor(self, value):
        if isinstance(value, (numpy.floating, numpy.ndarray)):
            return numpy.floor(value)
        assert 0, (value, type(value))  # unreachable

    def trunc(self, value):
        if isinstance(value, (numpy.floating, numpy.ndarray)):
            return numpy.trunc(value)
        assert 0, (value, type(value))  # unreachable

    def round(self, value):
        if isinstance(value, (numpy.floating, numpy.ndarray)):
            return numpy.round(value)
        assert 0, (value, type(value))  # unreachable

    def reciprocal(self, value):
        if isinstance(value, (numpy.floating, numpy.ndarray)):
            return numpy.reciprocal(value)
        assert 0, (value, type(value))  # unreachable

    def sqrt(self, value):
        if isinstance(value, (numpy.floating, numpy.ndarray)):
            return numpy.sqrt(value)
        assert 0, (value, type(value))  # unreachable

    def exp(self, value):
        if isinstance(value, (numpy.floating, numpy.ndarray)):
            return numpy.exp(value)
        assert 0, (value, type(value))  # unreachable

    def log(self, value):
        if isinstance(value, (numpy.floating, numpy.ndarray)):
            return numpy.log(value)
        assert 0, (value, type(value))  # unreachable

    def log1p(self, value):
        if isinstance(value, (numpy.floating, numpy.ndarray)):
            return numpy.log1p(value)
        assert 0, (value, type(value))  # unreachable

    def square(self, value):
        if isinstance(value, (numpy.floating, numpy.ndarray)):
            return numpy.square(value)
        assert 0, (value, type(value))  # unreachable

    def exp2(self, value):
        if isinstance(value, (numpy.floating, numpy.ndarray)):
            return numpy.exp2(value)
            if int(value) == value:
                return type(value)(2 ** int(value))
            return numpy.ldexp(type(value)(1), value)
        assert 0, (value, type(value))  # unreachable


def get_pi_over_two_multiword(dtype, prec=None, max_length=None):
    if prec is None:
        prec = {numpy.float16: 4, numpy.float32: 11, numpy.float64: 20}[dtype]
    max_prec = {numpy.float16: 24, numpy.float32: 149, numpy.float64: 1074}[dtype]
    ctx = mpmath.mp
    with ctx.workprec(max_prec):
        return mpf2multiword(dtype, ctx.pi / 2, p=prec, max_length=max_length)


def get_two_over_pi_multiword(dtype, prec=None, max_length=None):
    if prec is None:
        prec = {numpy.float16: 4, numpy.float32: 10, numpy.float64: 20}[dtype]
    max_prec = {numpy.float16: 24, numpy.float32: 149, numpy.float64: 1074}[dtype]
    ctx = mpmath.mp
    with ctx.workprec(max_prec):
        return mpf2multiword(dtype, 2 / ctx.pi, p=prec, max_length=max_length)


def show_ulp(ulp, title=None):
    rest = 0
    u5 = None
    if ulp and title is not None:
        print(f"{title}:")
    data = sum([[u] * ulp[u] for u in ulp], [])
    print(
        f"  ULP difference min/mean/median/max: {numpy.min(data)}/{numpy.mean(data):1.1f}"
        f"/{numpy.median(data):1.1f}/{numpy.max(data)}"
    )
    for i, u in enumerate(sorted(ulp)):
        if i < 5:
            print(f"  ULP difference {u}: {ulp[u]}")
        else:
            if u5 is None:
                u5 = u
            rest += ulp[u]
    else:
        if rest:
            if u5 == max(ulp):
                print(f"  ULP difference {u5}: {rest}")
            else:
                print(f"  ULP difference in [{u5}..{max(ulp)}]: {rest}")


def show_prec(prec, title=None):
    rest = 0
    u5 = None
    if prec and title is not None:
        print(f"{title}:")
    data = sum([[p] * prec[p] for p in prec], [])
    print(
        f"  precision min/mean/median/max: {numpy.min(data)}/{numpy.mean(data):1.1f}"
        f"/{numpy.median(data):1.1f}/{numpy.max(data)}"
    )
    lst = list(reversed(sorted(prec)))
    for i, u in enumerate(lst):
        if i < 4:
            print(f"  precision {u}: {prec[u]}")
        elif i == len(lst) - 5:
            rest += prec[u]
            if u5 is None:
                u5 = u
            if u5 == u:
                print(f"  precision {u}: {rest}")
            else:
                print(f"  precision in [{u}..{u5}]: {rest}")
            rest = 0
        elif i > len(lst) - 5:
            print(f"  precision {u}: {prec[u]}")
        else:
            if u5 is None:
                u5 = u
            rest += prec[u]
    else:
        if rest:
            print(f"  precision in [{min(prec)}..{u5}]: {rest}")


def ulp(x):
    """Return the unit in the last place of x.

    For finite x = m * 2 ** e, `ulp(x) == 2 ** e`.

    Invariants:
      ulp(inf) == inf
      ulp(-inf) == inf
      ulp(nan) == nan
      ulp(-x) == ulp(x)
      x + ulp(x) == nextafter(x, inf)  if x >= 0
      x - ulp(x) == nextafter(x, -inf)  if x < 0
    """
    dtype = type(x)
    if x == 0:
        return numpy.finfo(dtype).smallest_subnormal
    if numpy.isinf(x):
        return dtype("inf")
    if numpy.isnan(x):
        return dtype("nan")
    if x < 0:
        return ulp(-x)
    return numpy.ldexp(dtype(1), numpy.frexp(x)[1] + numpy.finfo(dtype).negep)


def overlapping(x, y):
    """Check if floating-point numbers are overlapping."""
    if x == y:
        return True
    if x == 0 or y == 0 or numpy.isnan(x) or numpy.isnan(y) or numpy.isnan(x):
        return False
    return abs(x) >= ulp(y) and abs(y) >= ulp(x)
