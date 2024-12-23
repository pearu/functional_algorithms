import platform
import pytest
import numpy
from functional_algorithms import fpu


def test_MXCSRRegister():
    if not fpu.MXCSRRegister.is_available():
        pytest.skip(f"N/A for {platform.machine()} platform")

    r = fpu.MXCSRRegister()

    with r(FZ=True):
        assert r.FZ

    with r(FZ=False):
        assert not r.FZ

    with r(DAZ=True):
        assert r.DAZ

    with r(DAZ=False):
        assert not r.DAZ

    fi = numpy.finfo(numpy.float32)
    s = fi.smallest_normal
    sub = fi.smallest_subnormal

    def check_ftz():
        return s * numpy.float32(0.5) == numpy.float32(0.0)

    def check_daz():
        return sub == numpy.float32(0.0)

    with r(FZ=False):
        assert not check_ftz()

    with r(FZ=True):
        assert check_ftz()

    with r(DAZ=False):
        assert not check_daz()

    with r(DAZ=True):
        assert check_daz()
