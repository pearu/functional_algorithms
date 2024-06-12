import numpy
import ctypes

libm = ctypes.cdll.LoadLibrary("libm.so.6")
libm.fma.argtypes = (ctypes.c_double, ctypes.c_double, ctypes.c_double)
libm.fma.restype = ctypes.c_double
libm.fmaf.argtypes = (ctypes.c_float, ctypes.c_float, ctypes.c_float)
libm.fmaf.restype = ctypes.c_float


def fma(x, y, z):
    if isinstance(x, numpy.float32) and isinstance(y, numpy.float32) and isinstance(y, numpy.float32):
        return numpy.float32(libm.fmaf(x, y, z))
    if isinstance(x, numpy.float64) and isinstance(y, numpy.float64) and isinstance(y, numpy.float64):
        return numpy.float64(libm.fma(x, y, z))
    return (x * y) + z


if __name__ == "__main__":
    cx, cy, cp, cq = map(numpy.float32, (0.70710677, 0.70710677, 0.9999999, 0.99999994))
    print(libm.fmaf(cx, cp, libm.fmaf(-cy, cq, 0)))
    print(fma(cx, cp, fma(-cy, cq, 0)))
