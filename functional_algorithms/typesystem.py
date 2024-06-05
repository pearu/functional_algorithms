import string


class Type:

    def __init__(self, context, kind, param):
        self.context = context
        assert kind in {"float", "complex", "integer", "boolean", "type"}, kind
        if isinstance(param, (int, type(None))):
            assert param in {1, 8, 16, 32, 64, 128, 256, 512, None}, param
        self.kind = kind
        self.param = param

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
            else:
                return cls(context, "type", obj)
                raise ValueError(f"expected string starting with float|complex|int|bool|type, got {obj}")
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
        elif isinstance(obj, Expr):
            return cls(context, "type", obj)

        raise NotImplementedError(type(obj))

    def __repr__(self):
        return f"Type({self.kind}, {self.param})"

    def __str__(self):
        if self.param is None:
            return self.kind
        if self.kind == "type":
            return str(self.param)
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

    def tostring(self, target):
        return target.Printer(dict()).tostring(self)

    def is_same(self, other):
        return (self.kind, self.bits) == (other.kind, other.bits)

    def is_same_kind(self, other):
        return self.kind == other.kind
