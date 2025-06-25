import numpy
from ..utils import vectorize_with_mpmath, vectorize_with_jax, vectorize_with_numpy


class mpmath_special_api:
    """scipy.special API interface to mpmath functions including
    workarounds to mpmath bugs.
    """

    def jv(self, v: float, z: float | complex):
        ctx = z.context
        return ctx.besselj(v, z)

    def j0(self, z: float | complex):
        ctx = z.context
        return ctx.besselj(0, z)

    def j1(self, z: float | complex):
        ctx = z.context
        return ctx.besselj(1, z)


class special_with_mpmath:
    """Namespace of scipy special functions on numpy arrays that use
    mpmath backend for evaluation and return numpy arrays as outputs.
    """

    _vfunc_cache = dict()

    def __init__(self, **params):
        self.params = params

    def __getattr__(self, name):
        name = dict().get(name, name)
        key = name, tuple(sorted(self.params.items()))
        if key in self._vfunc_cache:
            return self._vfunc_cache[key]
        if hasattr(mpmath_special_api, name):
            vfunc = vectorize_with_mpmath(getattr(mpmath_special_api(), name), **self.params)
            self._vfunc_cache[key] = vfunc
            return vfunc
        raise NotImplementedError(f"vectorize_with_mpmath.{name}")


class special_with_scipy:
    """Namespace of scipy special functions on numpy arrays that use
    scipy.special backend for evaluation and return numpy arrays as
    outputs.
    """

    _vfunc_cache = dict()

    def __init__(self, **params):
        self.params = params
        self.params.pop("device")
        self.params.pop("dtype")

    def __getattr__(self, name):
        name = dict().get(name, name)
        key = name, tuple(sorted(self.params.items()))
        if key in self._vfunc_cache:
            return self._vfunc_cache[key]
        import scipy.special as sp

        vfunc = vectorize_with_numpy(getattr(sp, name), **self.params)
        self._vfunc_cache[key] = vfunc
        return vfunc


class special_with_jax:
    """Namespace of scipy special functions on numpy arrays that use
    jax.scipy.special backend for evaluation and return numpy arrays
    as outputs.
    """

    _vfunc_cache = dict()

    def __init__(self, **params):
        self.params = params

    def __getattr__(self, name):
        name = dict().get(name, name)
        key = name, tuple(sorted(self.params.items()))
        if key in self._vfunc_cache:
            return self._vfunc_cache[key]
        import jax

        vfunc = vectorize_with_jax(getattr(jax.scipy.special, name), **self.params)
        self._vfunc_cache[key] = vfunc
        return vfunc


def function_validation_parameters(func_name, dtype, device=None):
    if isinstance(dtype, str):
        dtype_name = dtype
    elif isinstance(dtype, numpy.dtype):
        dtype_name = dtype.type.__name__
    else:
        dtype_name = dtype.__name__

    real_dtype = dict(float32=numpy.float32, float64=numpy.float64, complex64=numpy.float32, complex128=numpy.float64)[
        dtype_name
    ]

    # If a function has symmetries, exclude superfluous samples by
    # specifying a region of the function domain:
    samples_limits = dict(
        min_real_value=-real_dtype(numpy.inf),
        max_real_value=real_dtype(numpy.inf),
        min_imag_value=-real_dtype(numpy.inf),
        max_imag_value=real_dtype(numpy.inf),
    )

    # ulp_diff(func(sample), reference(sample)) <= max_valid_ulp_count
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

    if func_name == "j0":
        """
        For large x, j0(x) ~ sqrt(2/pi/x) * (cos(x - pi/4) + O(1/x)).

        j0(x_max) ~ sqrt(2/pi/x_max) ~ smallest_normal
        x_max ~ smallest_normal ** -2 * pi / 2
        """
        max_valid_ulp_count = 10
        extra_prec_multiplier = 50
        samples_limits["min_real_value"] = -1e12
        samples_limits["max_real_value"] = 1e12
        samples_limits["min_imag_value"] = samples_limits["min_real_value"]
        samples_limits["max_imag_value"] = samples_limits["max_real_value"]

    return dict(
        extra_prec_multiplier=extra_prec_multiplier,
        max_valid_ulp_count=max_valid_ulp_count,
        max_bound_ulp_width=max_bound_ulp_width,
        samples_limits=samples_limits,
    )
