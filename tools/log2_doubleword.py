import functional_algorithms as fa
import mpmath
import numpy


def main():
    dtype = numpy.float16

    workprec = 1000
    highprec = 2 * workprec
    max_p = {numpy.float16: 9, numpy.float32: 19, numpy.float64: 45}[dtype]
    min_p = {numpy.float16: 5, numpy.float32: 11, numpy.float64: 26}[dtype]
    bitwidth = {numpy.float16: 16, numpy.float32: 32, numpy.float64: 64}[dtype]
    bytesize = {numpy.float16: 2, numpy.float32: 4, numpy.float64: 8}[dtype]
    kmax = 2 * bytesize**3

    with mpmath.workprec(highprec):
        log2_hp = mpmath.log(2)
        ln2inv = fa.utils.mpf2float(numpy.longdouble, 1 / log2_hp)
        ln2half = fa.utils.mpf2float(numpy.longdouble, log2_hp / 2)
        print(f"{ln2inv=}")
        # print(f"{str(1 / log2_hp)[:30]=}")
        print(f"{ln2half=}")

    with mpmath.mp.workprec(workprec):

        for p in range(min_p, max_p + 1):
            hi, lo = fa.utils.mpf2multiword(dtype, mpmath.log(2), p=p, max_length=2)
            with mpmath.mp.workprec(highprec):
                ctx = mpmath.mp

                ae = dtype(0)
                for k in range(1, kmax + 1):
                    ae = max(
                        ae,
                        fa.utils.mpf2float(
                            dtype, abs((fa.utils.float2mpf(ctx, hi * k) + fa.utils.float2mpf(ctx, lo * k)) - k * mpmath.log(2))
                        ),
                    )

                abserr = fa.utils.mpf2float(
                    dtype, abs((fa.utils.float2mpf(ctx, hi) + fa.utils.float2mpf(ctx, lo)) - mpmath.log(2))
                )
                print(f"{p=} {abserr=} {ae=} {hi=} {lo=}")


if __name__ == "__main__":
    main()
