import string
import numpy


class Type:

    def __new__(cls, context, kind, param):
        obj = object.__new__(cls)
        obj._initialize(context, kind, param)

        # Type instances are singletons within the given context:
        if obj in context._types:
            return context._types[obj]
        context._types[obj] = obj

        return obj

    def _initialize(self, context, kind, param):
        self.context = context
        assert kind in {"float", "complex", "integer", "boolean", "type", "list", "array"}, kind
        if isinstance(param, (int, type(None))):
            assert param in {1, 8, 16, 32, 64, 128, 256, 512, None}, param
        elif isinstance(param, tuple):
            if kind == "array":
                assert len(param) <= 2
                if len(param) > 0:
                    assert isinstance(param[0], type(self))
                if len(param) == 2:
                    assert isinstance(param[1], (tuple, type(None)))
            else:
                for p in param:
                    assert isinstance(p, type(self))
        elif isinstance(param, type(self)):
            if kind == "array":
                param = (param,)
        self.kind = kind
        self.param = param

    def __hash__(self):
        return hash((self.kind, self.param))

    def __eq__(self, other):
        if self is other:
            return True
        if isinstance(other, type(self)):
            return self.context is other.context and self.kind == other.kind and self.param == other.param
        return False

    @property
    def bits(self):
        if self.kind in {"float", "complex", "integer", "boolean"}:
            return self.param
        raise NotImplementedError(self)

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
                obj = obj[len(kind) :].lstrip()
                assert obj.startswith("[") and obj.endswith("]")
                obj = obj[1:-1].strip()
                params = tuple(cls.fromobject(context, s.strip()) for s in obj.split(","))
                return cls(context, kind, params)
            elif obj.lower().startswith("array"):
                kind = "array"
                obj = obj[len(kind) :].lstrip()
                if obj.startswith("["):
                    i = obj.index("]")
                    assert i > 0, obj
                    itype = cls.fromobject(context, obj[1:i].strip())
                    obj = obj[i + 1].lstrip()
                    assert len(obj) == 0, obj  # parsing dims not implemented
                    params = (itype,)
                else:
                    if obj.lower().startswith("like"):
                        obj = obj[4:].lstrip()
                    assert len(obj) == 0, obj  # parsing dims not implemented
                    params = None
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
        # TODO: array from ndarray
        raise NotImplementedError((obj, type(obj)))

    def __repr__(self):
        return f"Type({self.kind}, {self.param})"

    def __str__(self):
        if self.param is None:
            return self.kind
        if self.kind == "type":
            return str(self.param)
        if self.kind == "list":
            return f"{self.kind}[{', '.join(map(str, self.param))}]"
        if self.kind == "array":
            if self.param is None or not self.param:
                return kind
            assert len(self.param) == 1, self.param  # shape support not impl
            itype = self.param[0]
            return f"{self.kind}[{itype}]"
        return f"{self.kind}{self.param}"

    def max(self, other):
        if self == other:
            return self
        if self.kind == "array":
            if other.kind == "array":
                if self.param is not None:
                    if other.param is not None:
                        itype1 = self.param[0]
                        itype2 = other.param[0]
                        return type(self)(self.kind, itype1.max(itype2))
                    else:
                        return other
                return self
            assert other.kind in {"integer", "boolean", "float", "complex"}, other.kind
            return self
        elif other.kind == "array":
            return other.max(self)
        if "complex" in {self.kind, other.kind}:
            kind = "complex"
        elif "float" in {self.kind, other.kind}:
            kind = "float"
        elif "integer" in {self.kind, other.kind}:
            kind = "integer"
        elif "boolean" in {self.kind, other.kind}:
            kind = "boolean"
        elif self.kind == "list" and other.kind == "list":
            if len(self.param) < len(other.param):
                assert self.param == other.param[: len(self.param)]
                return other
            else:
                assert self.param[: len(other.param)] == other.param
                return self
        else:
            raise NotImplementedError((self.kind, other.kind))
        # TODO: array type support
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

    @property
    def is_array(self):
        return self.kind == "array"

    def tostring(self, target):
        return target.Printer(dict()).tostring(self)

    def is_same(self, other):
        if self is other:
            return True
        if self.kind == other.kind:
            if self.kind == "list":
                if len(self.param) == len(other.param):
                    for i1, i2 in zip(self.param, other.param):
                        if not i1.is_same(i2):
                            return False
                    return True
            elif self.kind == "array":
                return self.param == other.param
            else:
                return self.param == other.param
        return False

    def is_same_kind(self, other):
        return self.kind == other.kind

    def asdtype(self):
        if self.kind == "list":
            return [item.asdtype() for item in self.param]
        assert self.kind != "array"  # not impl
        return getattr(numpy, str(self), None)
