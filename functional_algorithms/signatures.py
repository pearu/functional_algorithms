"""Provides signatures of math functions.
"""


def all_signatures():
    yield from _signature_iter(_array_api_element_wise_functions)


def filter_signatures(atypes="C", rtypes="C"):
    atypes = tuple(atypes)
    rtypes = tuple(rtypes)
    for sig in all_signatures():
        if sig["atypes"] == atypes and sig["rtypes"] == rtypes:
            yield sig


def _signature_iter(text):
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#") or not line:
            continue
        left, rtype = line.split("->", 1)
        rtypes = tuple(r.strip() for r in rtype.strip().split(","))
        name = line[: line.find("(")].strip()
        atypes = []
        anames = []
        for arg in left.rstrip()[left.find("(") + 1 : -1].split(","):
            arg = arg.strip()
            if arg == "/":
                break
            atype, aname = arg.split()
            atype = atype.strip()
            aname = aname.strip()
            atypes.append(atype)
            anames.append(anames)
        yield dict(name=name, atypes=tuple(atypes), anames=tuple(anames), rtypes=rtypes)


_array_api_element_wise_functions = """
abs(F x, /) -> F
abs(C x, /) -> F
acos(F x, /) -> F
acos(C x, /) -> C
acosh(F x, /) -> F
acosh(C x, /) -> C
add(F x1, F x2, /) -> F
add(C x1, C x2, /) -> C
add(F x1, C x2, /) -> C
add(C x1, V x2, /) -> C
asin(F x, /) -> F
asin(C x, /) -> C
asinh(F x, /) -> F
asinh(C x, /) -> C
atan(F x, /) -> F
atan(C x, /) -> C
atan2(F x1, F x2, /) -> F
atanh(F x, /) -> F
atanh(C x, /) -> C
conj(C x, /) -> C
cos(F x, /) -> F
cos(C x, /) -> C
cosh(F x, /) -> F
cosh(C x, /) -> C
exp(F x, /) -> F
exp(C x, /) -> C
expm1(F x, /) -> F
expm1(C x, /) -> C
hypot(F x1, F x2, /) -> F
imag(C x, /) -> F
log(F x, /) -> F
log(C x, /) -> C
log1p(F x, /) -> F
log1p(C x, /) -> C
log2(F x, /) -> F
log2(C x, /) -> C
log10(F x, /) -> F
log10(C x, /) -> C
logaddexp(F x1, F x2, /) -> F
multiply(F x1, F x2, /) -> F
multiply(C x1, C x2, /) -> C
multiply(F x1, C x2, /) -> C
multiply(C x1, F x2, /) -> C
negative(F x, /) -> F
negative(C x, /) -> C
positive(F x, /) -> F
positive(C x, /) -> C
pow(F x1, F x2, /) -> F
pow(C x1, C x2, /) -> C
pow(C x1, F x2, /) -> C
pow(F x1, C x2, /) -> C
real(C x, /) -> F
sin(F x, /) -> F
sin(C x, /) -> C
sinh(F x, /) -> F
sinh(C x, /) -> C
square(F x, /) -> F
square(C x, /) -> C
sqrt(F x, /) -> F
sqrt(C x, /) -> C
subtract(F x1, F x2, /) -> F
subtract(C x1, C x2, /) -> C
subtract(F x1, C x2, /) -> C
subtract(C x1, V x2, /) -> C
tan(F x, /) -> F
tan(C x, /) -> C
tanh(F x, /) -> F
tanh(C x, /) -> C
"""
