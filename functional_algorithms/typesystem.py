import string
import numpy


class Type:

    def __init__(self, context, kind, param):
        self.context = context
        assert kind in {"float", "complex", "integer", "boolean", "type", "list"}, kind
        if isinstance(param, (int, type(None))):
            assert param in {1, 8, 16, 32, 64, 128, 256, 512, None}, param
        elif isinstance(param, tuple):
            for p in param:
                assert isinstance(p, type(self))
        self.kind = kind
        self.param = param

    def __hash__(self):
        return hash((self.kind, self.param))

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self.context is other.context and self.kind == other.kind and self.param == other.param
        return False

    @property
    def bits(self):
        if self.kind in {"float", "complex", "integer", "boolean"}:
            return self.param
        raise NotImplementedError(self.kind)

    @classmethod
    def fromobject(cls, context, obj):
        from .expr import Expr

        if isinstance(obj, str):
            if obj.startswith("float"):
                kind = "float"
            elif obj.startswith("int"):
                kind = "integer"
            elif obj.startswith("complex"):
                kind = "complex"
            elif obj.startswith("bool"):
                kind = "boolean"
            elif obj.startswith("type"):
                kind = "type"
            elif obj.startswith("list"):
                kind = "list"
                obj = obj[4:].lstrip()
                assert obj.startswith("[") and obj.endswith("]")
                obj = obj[1:-1].strip()
                params = tuple(cls.fromobject(context, s.strip()) for s in obj.split(","))
                return cls(context, kind, params)
            else:
                return cls(context, "type", obj)
            bits = obj.strip(string.ascii_letters) or None
            if bits:
                bits = int(bits)
            return cls(context, kind, bits)
        elif isinstance(obj, cls):
            return obj
        elif obj is float:
            # notice that `float` interpretted as `"float"`. Use
            # `"float64"` to specify the bitwidth of the type.
            return cls(context, "float", None)
        elif obj is complex:
            return cls(context, "complex", None)
        elif obj is int:
            return cls(context, "integer", None)
        elif obj is bool:
            return cls(context, "boolean", None)
        elif obj is list:
            return cls(context, "list", None)
        elif isinstance(obj, Expr):
            return cls(context, "type", obj)
        elif issubclass(obj, numpy.number):
            return cls.fromobject(context, obj.__name__)
        elif isinstance(obj, list):
            return cls(context, "list", tuple(cls.fromobject(context, item) for item in obj))
        # TODO: list from typing.List
        raise NotImplementedError(type(obj))

    def __repr__(self):
        return f"Type({self.kind}, {self.param})"

    def __str__(self):
        if self.param is None:
            return self.kind
        if self.kind == "type":
            return str(self.param)
        if self.kind == "list":
            return f"{self.kind}[{', '.join(map(str, self.param))}]"
        return f"{self.kind}{self.param}"

    def max(self, other):
        if self == other:
            return self
        if "complex" in {self.kind, other.kind}:
            kind = "complex"
        elif "float" in {self.kind, other.kind}:
            kind = "float"
        elif "integer" in {self.kind, other.kind}:
            kind = "integer"
        elif "boolean" in {self.kind, other.kind}:
            kind = "boolean"
        else:
            raise NotImplementedError((self.kind, other.kind))
        bits = max([t.bits for t in [self, other] if t.kind == kind and t.bits is not None] or [None])
        return type(self)(self.context, kind, bits)

    @property
    def complex_part(self):
        assert self.kind == "complex", self
        bits = self.bits // 2 if self.bits is not None else None
        return type(self)(self.context, "float", bits)

    @property
    def is_float(self):
        return self.kind == "float"

    @property
    def is_complex(self):
        return self.kind == "complex"

    @property
    def is_integer(self):
        return self.kind == "integer"

    @property
    def is_boolean(self):
        return self.kind == "boolean"

    @property
    def is_type(self):
        return self.kind == "type"

    @property
    def is_list(self):
        return self.kind == "list"

    def tostring(self, target):
        return target.Printer(dict()).tostring(self)

    def is_same(self, other):
        if self.kind == other.kind:
            if self.kind == "list":
                if len(self.params) == len(other.params):
                    for i1, i2 in zip(self.params, other.params):
                        if not i1.is_same(i2):
                            return False
                    return True
            else:
                return (self.kind, self.bits) == (other.kind, other.bits)
        return False

    def is_same_kind(self, other):
        return self.kind == other.kind

    def asdtype(self):
        if self.kind == "list":
            return [item.asdtype() for item in self.param]
        return getattr(numpy, str(self), None)
