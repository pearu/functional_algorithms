import numpy
import cProfile
import pstats
import io
import functional_algorithms as fa
import functional_algorithms.floating_point_algorithms as fpa


def square_of_abs(ctx, z: complex):
    x = ctx.real(z)
    y = ctx.imag(z)
    largest = ctx.constant("largest", x).reference("largest")
    C = fpa.get_veltkamp_splitter_constant(ctx, largest)
    Q, P = fpa.get_is_power_of_two_constants(ctx, largest)
    three_over_two = ctx.constant(1.5, x)
    return fpa.dot2(ctx, x, x, y, y, C, Q, P, three_over_two)


def test_dot2():
    dtype = numpy.complex64
    ctx = fa.Context(paths=[fa.algorithms])
    with cProfile.Profile() as pr:
        graph = ctx.trace(square_of_abs, dtype)
        graph2 = graph.rewrite(fa.targets.numpy, fa.rewrite)
        func = fa.targets.numpy.as_function(graph2)
        assert func is not None

        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
        ps.print_stats()
        print(s.getvalue())
