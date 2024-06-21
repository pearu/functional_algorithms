import numpy
import mpmath
import warnings
from collections import defaultdict

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda seq: seq


class UNSPECIFIED:

    def __str__(self):
        return type(self).__name__

    __repr__ = __str__


UNSPECIFIED = UNSPECIFIED()
default_flush_subnormals = False


class vectorize_with_mpmath(numpy.vectorize):
    """Same as numpy.vectorize but using mpmath backend for function evaluation."""

    map_float_to_complex = dict(
        float16="complex32", float32="complex64", float64="complex128", float128="complex256", longdouble="clongdouble"
    )
    map_complex_to_float = {v: k for k, v in map_float_to_complex.items()}

    float_prec = dict(
        # float16=11,
        float32=24,
        float64=53,
        # float128=113,
        # longdouble=113
    )

    float_subexp = dict(float16=-23, float32=-148, float64=-1073, float128=-16444)
    float_minexp = dict(float16=-13, float32=-125, float64=-1021, float128=-16381)

    float_maxexp = dict(
        float16=16,
        float32=128,
        float64=1024,
        float128=16384,
    )

    def __init__(self, *args, **kwargs):
        self.extra_prec_multiplier = kwargs.pop("extra_prec_multiplier", 0)
        self.extra_prec = kwargs.pop("extra_prec", 0)
        flush_subnormals = kwargs.pop("flush_subnormals", UNSPECIFIED)
        self.flush_subnormals = flush_subnormals if flush_subnormals is UNSPECIFIED else default_flush_subnormals
        self.contexts = dict()
        self.contexts_inv = dict()
        for fp_format, prec in self.float_prec.items():
            ctx = mpmath.mp.clone()
            ctx.prec = prec
            self.contexts[fp_format] = ctx
            self.contexts_inv[ctx] = fp_format

        super().__init__(*args, **kwargs)

    def get_context(self, x):
        if isinstance(x, (numpy.ndarray, numpy.floating, numpy.complexfloating)):
            fp_format = str(x.dtype)
            fp_format = self.map_complex_to_float.get(fp_format, fp_format)
            return self.contexts[fp_format]
        raise NotImplementedError(f"get mpmath context from {type(x).__name__} instance")

    def nptomp(self, x):
        """Convert numpy array/scalar to an array/instance of mpmath number type."""
        if isinstance(x, numpy.ndarray):
            return numpy.fromiter(map(self.nptomp, x.flatten()), dtype=object).reshape(x.shape)
        elif isinstance(x, numpy.floating):
            ctx = self.get_context(x)
            prec, rounding = ctx._prec_rounding
            if numpy.isposinf(x):
                return ctx.make_mpf(mpmath.libmp.finf)
            elif numpy.isneginf(x):
                return ctx.make_mpf(mpmath.libmp.fninf)
            elif numpy.isnan(x):
                return ctx.make_mpf(mpmath.libmp.fnan)
            elif numpy.isfinite(x):
                mantissa, exponent = numpy.frexp(x)
                man = int(numpy.ldexp(mantissa, prec))
                exp = int(exponent - prec)
                r = ctx.make_mpf(mpmath.libmp.from_man_exp(man, exp, prec, rounding))
                assert ctx.isfinite(r), r._mpf_
                return r
        elif isinstance(x, numpy.complexfloating):
            re, im = self.nptomp(x.real), self.nptomp(x.imag)
            return re.context.make_mpc((re._mpf_, im._mpf_))
        raise NotImplementedError(f"convert {type(x).__name__} instance to mpmath number type")

    def mptonp(self, x):
        """Convert mpmath instance to numpy array/scalar type."""
        if isinstance(x, numpy.ndarray) and x.dtype.kind == "O":
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
                if ctx.isfinite(x):
                    sign, man, exp, bc = mpmath.libmp.normalize(*x._mpf_, *ctx._prec_rounding)
                    assert bc >= 0, (sign, man, exp, bc, x._mpf_)
                    zexp = self.float_minexp[fp_format] if self.flush_subnormals else self.float_subexp[fp_format]
                    if exp + bc < zexp:
                        return -ctx.zero if sign else ctx.zero
                    if exp + bc > self.float_maxexp[fp_format]:
                        return ctx.ninf if sign else ctx.inf
                    man = dtype(-man if sign else man)
                    r = numpy.ldexp(man, exp)
                    assert numpy.isfinite(r), (x, r, x._mpf_, man)
                    return r
                elif ctx.isnan(x):
                    return dtype(numpy.nan)
                elif ctx.isinf(x):
                    return dtype(-numpy.inf if x._mpf_[0] else numpy.inf)
        raise NotImplementedError(f"convert {type(x)} instance to numpy floating point type")

    def __call__(self, *args, **kwargs):
        mp_args = []
        context = None
        for a in args:
            if isinstance(a, (numpy.ndarray, numpy.floating, numpy.complexfloating)):
                mp_args.append(self.nptomp(a))
                if context is None:
                    context = self.get_context(a)
                else:
                    assert context is self.get_context(a), (context, self.get_context(a), a.dtype)
            else:
                mp_args.append(a)

        extra_prec = int(context.prec * self.extra_prec_multiplier) + self.extra_prec
        with context.extraprec(extra_prec):
            result = super().__call__(*mp_args, **kwargs)

        if isinstance(result, tuple):
            lst = []
            for r in result:
                if (isinstance(r, numpy.ndarray) and r.dtype.kind == "O") or isinstance(r, mpmath.ctx_mp.mpnumeric):
                    r = self.mptonp(r)
                lst.append(r)
            return tuple(lst)

        if (isinstance(result, numpy.ndarray) and result.dtype.kind == "O") or isinstance(result, mpmath.ctx_mp.mpnumeric):
            return self.mptonp(result)

        return result


class numpy_with_mpmath:
    """Namespace of universal functions on numpy arrays that use mpmath
    backend for evaluation and return numpy arrays as outputs.
    """

    _provides = [
        "abs",
        "absolute",
        "sqrt",
        "exp",
        "expm1",
        "exp2",
        "log",
        "log1p",
        "log10",
        "log2",
        "sin",
        "cos",
        "tan",
        "arcsin",
        "arccos",
        "arctan",
        "sinh",
        "cosh",
        "tanh",
        "arcsinh",
        "arccosh",
        "arctanh",
        "square",
        "positive",
        "negative",
        "conjugate",
        "sign",
        "sinc",
        "normalize",
        "hypot",
    ]

    _mpmath_names = dict(
        abs="absmin",
        absolute="absmin",
        log="ln",
        arcsin="asin",
        arccos="acos",
        arctan="atan",
        arcsinh="asinh",
        arccosh="acosh",
        arctanh="atanh",
    )

    def __init__(self, extra_prec_multiplier=0, extra_prec=0, flush_subnormals=UNSPECIFIED):

        for name in self._provides:
            mp_name = self._mpmath_names.get(name, name)

            if hasattr(self, name):
                op = getattr(self, name)
            else:
                if name in {"hypot"}:
                    # binary ops
                    def op(x, y, mp_name=mp_name):
                        return getattr(x.context, mp_name)(x, y)

                else:
                    # unary ops
                    def op(x, mp_name=mp_name):
                        return getattr(x.context, mp_name)(x)

            vop = vectorize_with_mpmath(
                op, extra_prec_multiplier=extra_prec_multiplier, extra_prec=extra_prec, flush_subnormals=flush_subnormals
            )
            setattr(self, name, vop)
            if not hasattr(self, mp_name):
                setattr(self, mp_name, vop)

    # The following function methods operate on mpmath number instances.
    # The corresponding function names must be listed in
    # numpy_with_mpmath._provides list.

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

    def square(self, x):
        ctx = x.context
        if isinstance(x, ctx.mpc):
            if abs(x.real) == abs(x.imag):
                return ctx.make_mpc((ctx.zero._mpf_, (x.real * x.imag * 2)._mpf_))
            return ctx.make_mpc((((x.real - x.imag) * (x.real + x.imag))._mpf_, (x.real * x.imag * 2)._mpf_))
        return x * x

    def positive(self, x):
        return x

    def negative(self, x):
        return -x

    def sqrt(self, x):
        ctx = x.context
        if isinstance(x, ctx.mpc):
            # Workaround mpmath 1.3 bug in sqrt(+-inf+-infj) evaluation (see mpmath/mpmath#776).
            if ctx.isinf(x.imag):
                return ctx.make_mpc((ctx.inf._mpf_, x.imag._mpf_))
        return ctx.sqrt(x)

    def expm1(self, x):
        return x.context.expm1(x)

    def log1p(self, x):
        ctx = x.context
        if isinstance(x, ctx.mpc):
            # Workaround mpmath 1.3 bug in log(+-inf+-infj) evaluation (see mpmath/mpmath#774).
            if ctx.isinf(x.real) and ctx.isinf(x.imag):
                pi = ctx.pi
            if x.real > 0 and x.imag > 0:
                return ctx.make_mpc((x.real._mpf_, (pi / 4)._mpf_))
            if x.real > 0 and x.imag < 0:
                return ctx.make_mpc((x.real._mpf_, (-pi / 4)._mpf_))
            if x.real < 0 and x.imag < 0:
                return ctx.make_mpc(((-x.real)._mpf_, (-3 * pi / 4)._mpf_))
            if x.real < 0 and x.imag > 0:
                return ctx.make_mpc(((-x.real)._mpf_, (3 * pi / 4)._mpf_))
        return ctx.log1p(x)

    def tan(self, x):
        ctx = x.context
        if isinstance(x, ctx.mpc):
            # Workaround mpmath 1.3 bug in tan(+-inf+-infj) evaluation (see mpmath/mpmath#781).
            if ctx.isinf(x.imag) and (ctx.isinf(x.real) or ctx.isfinite(x.real)):
                if x.imag > 0:
                    return ctx.make_mpc((ctx.zero._mpf_, ctx.one._mpf_))
                return ctx.make_mpc((ctx.zero._mpf_, (-ctx.one)._mpf_))
            if ctx.isinf(x.real) and ctx.isfinite(x.imag):
                return ctx.make_mpc((ctx.nan._mpf_, ctx.nan._mpf_))
        return ctx.tan(x)

    def tanh(self, x):
        ctx = x.context
        if isinstance(x, ctx.mpc):
            # Workaround mpmath 1.3 bug in tanh(+-inf+-infj) evaluation (see mpmath/mpmath#781).
            if ctx.isinf(x.imag) and (ctx.isinf(x.real) or ctx.isfinite(x.real)):
                if x.imag > 0:
                    return ctx.make_mpc((ctx.zero._mpf_, ctx.one._mpf_))
                return ctx.make_mpc((ctx.zero._mpf_, (-ctx.one)._mpf_))
            if ctx.isinf(x.real) and ctx.isfinite(x.imag):
                return ctx.make_mpc((ctx.nan._mpf_, ctx.nan._mpf_))
        return ctx.tanh(x)

    def log2(self, x):
        return x.context.ln(x) / x.context.ln2

    def log10(self, x):
        return x.context.ln(x) / x.context.ln10

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


def real_samples(
    size=10,
    dtype=numpy.float32,
    include_infinity=True,
    include_zero=True,
    include_subnormal=False,
    include_nan=False,
    include_huge=True,
    nonnegative=False,
):
    """Return a 1-D array of real line samples.

    Parameters
    ----------
    size : int
      Initial size of the samples array. A minimum value is 6. The
      actual size of the returned array may differ from size.
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
    """
    if isinstance(dtype, str):
        dtype = getattr(numpy, dtype)
    assert dtype in {numpy.float32, numpy.float64}, dtype
    fi = numpy.finfo(dtype)
    start = fi.minexp + fi.negep + 1 if include_subnormal else fi.minexp
    end = fi.maxexp
    num = size // 2 if not nonnegative else size
    if include_infinity:
        num -= 1
    with warnings.catch_warnings(action="ignore"):
        finite_positive = numpy.logspace(start, end, base=2, num=num, dtype=dtype)

    finite_positive[-1] = fi.max
    if include_huge and size > 7:
        finite_positive[-2] = -numpy.nextafter(-fi.max, numpy.inf, dtype=dtype)
    parts = []
    if not nonnegative:
        if include_infinity:
            parts.append(numpy.array([-numpy.inf], dtype=dtype))
        parts.append(-finite_positive[::-1])
    if include_zero:
        parts.append(numpy.array([0], dtype=dtype))
    parts.append(finite_positive)
    if include_infinity:
        parts.append(numpy.array([numpy.inf], dtype=dtype))
    if include_nan:
        parts.append(numpy.array([numpy.nan], dtype=dtype))
    return numpy.concatenate(parts)


def complex_samples(
    size=(10, 10),
    dtype=numpy.float32,
    include_infinity=True,
    include_zero=True,
    include_subnormal=False,
    include_nan=False,
    include_huge=True,
    nonnegative=False,
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
        nonnegative=nonnegative,
        include_huge=include_huge,
    )
    s2 = complex_samples(
        size=size[1],
        dtype=dtype,
        include_infinity=include_infinity,
        include_zero=include_zero,
        include_subnormal=include_subnormal,
        include_nan=include_nan,
        nonnegative=nonnegative,
        include_huge=include_huge,
    )
    shape1 = s1.shape
    shape2 = s2.shape
    s1, s2 = numpy.tile(s1, shape2), s2.repeat(shape1[0], 0).repeat(shape1[1], 1)
    return s1, s2


def iscomplex(value):
    return isinstance(value, (complex, numpy.complexfloating))


def isfloat(value):
    return isinstance(value, (float, numpy.floating))


def diff_ulp(x, y, flush_subnormals=UNSPECIFIED) -> int:
    """Return ULP distance between two floating point numbers.

    For complex inputs, return largest ULP among real and imaginary
    parts.

    When flush_subnormals is set to True, ULP difference does not
    account for subnormals while subnormal values are rounded to
    nearest normal, ties to even.
    """
    if isinstance(x, numpy.floating):
        uint = {numpy.float64: numpy.uint64, numpy.float32: numpy.uint32, numpy.float16: numpy.uint16}[x.dtype.type]
        sx = -1 if x <= 0 else 1
        sy = -1 if y <= 0 else 1
        x, y = abs(x), abs(y)
        ix, iy = int(x.view(uint)), int(y.view(uint))
        if numpy.isfinite(x) and numpy.isfinite(y):
            flush_subnormals if flush_subnormals is UNSPECIFIED else default_flush_subnormals
            if flush_subnormals:
                fi = numpy.finfo(x.dtype)
                i = int(fi.smallest_normal.view(uint)) - 1  # 0 distance to largest subnormal
                ix = ix - i if ix > i else (0 if 2 * ix <= i else 1)
                iy = iy - i if iy > i else (0 if 2 * iy <= i else 1)
            if sx != sy:
                # distance is measured through 0 value
                return ix + iy
            return ix - iy if ix >= iy else iy - ix
        elif ix == iy and sx == sy:
            return 0
        return {numpy.float64: 2**64, numpy.float32: 2**32, numpy.float16: 2**16}[x.dtype.type]
    elif isinstance(x, numpy.complexfloating):
        return max(
            diff_ulp(x.real, y.real, flush_subnormals=flush_subnormals),
            diff_ulp(x.imag, y.imag, flush_subnormals=flush_subnormals),
        )
    raise NotImplementedError(type(x))


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


def validate_function(func, reference, samples, dtype, verbose=True, flush_subnormals=UNSPECIFIED, enable_progressbar=False):
    fi = numpy.finfo(dtype)
    ulp_stats = defaultdict(int)
    for sample in tqdm(samples) if enable_progressbar else samples:
        if isinstance(sample, tuple):
            v1 = func(*sample)
            v2 = reference(*sample)
        else:
            v1 = func(sample)
            v2 = reference(sample)

        v2 = v2[()]
        assert v1.dtype == v2.dtype, (v1, v2)
        ulp = diff_ulp(v1, v2, flush_subnormals=flush_subnormals)
        ulp_stats[ulp] += 1
        if ulp > 2 and verbose:
            print(f"--> {sample, v1, v2, ulp=}")
        if ulp >= 3:
            print(f"--> {sample, v1, v2, ulp=}")

    return ulp_stats[-1] == 0 and max(ulp_stats) <= 3, ulp_stats


def format_python(code):
    try:
        import black
    except ImportError:
        return code
    return black.format_str(code, mode=black.FileMode(line_length=127))


def format_cpp(code):
    import os
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
