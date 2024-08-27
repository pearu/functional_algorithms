import numpy
from collections import defaultdict


class TextImage:
    """Image for text terminals."""

    def __init__(self, rows=0, cols=0, bg=" "):
        """
        Parameters
        ----------
        rows: int
          Initial number of image rows, corresponds to image height.
        cols: int
          Initial number of columns, corresponds to image width.
        bg: str
          Single character used as the background of the image.
        """
        self.bg = bg
        self.table = numpy.full((rows, cols), bg, dtype="c")

    def _allocate(self, row, col):
        """Re-allocate table so that given table cell is accessible.

        This corresponds to expanding image to right or down.

        Notice that for an originally empty table, the shape of
        resulting table will be (row + 1, col + 1).
        """
        if row >= self.table.shape[0]:
            self.table = numpy.vstack(
                (self.table, numpy.full((row - self.table.shape[0] + 1, self.table.shape[1]), self.bg, dtype=self.table.dtype))
            )
        if col >= self.table.shape[1]:
            self.table = numpy.hstack(
                (self.table, numpy.full((self.table.shape[0], col - self.table.shape[1] + 1), self.bg, dtype=self.table.dtype))
            )

    def _handle_negative_location(self, row, col):
        while row < 0:
            row += self.table.shape[0]
        while col < 0:
            col += self.table.shape[1]
        return row, col

    def _shift_by_loc(self, row, col, loc, shape):
        drow = dcol = 0
        if loc in {"ul", "l", "ll"}:
            pass
        elif loc in {"uc", "c", "lc"}:
            dcol = -(shape[1] // 2)
        elif loc in {"ur", "r", "lr"}:
            dcol = -shape[1]
        else:
            assert 0, loc  # unreachable
        if loc in {"ul", "uc", "ur"}:
            pass
        elif loc in {"l", "c", "r"}:
            drow = -(shape[0] // 2)
        elif loc in {"ll", "lc", "lr"}:
            drow = -shape[0]
        else:
            assert 0, loc  # unreachable
        return row + drow, col + dcol

    @staticmethod
    def _value_to_text(value):
        if isinstance(value, numpy.ndarray):
            dtype = value.dtype
            assert value.shape == (), value.shape
        elif isinstance(value, numpy.floating):
            dtype = type(value)
        else:
            assert 0, type(value)  # not impl
        f = numpy.finfo(dtype)
        if numpy.isnan(value):
            return "nan"
        if numpy.isposinf(value):
            return "+inf"
        if numpy.isneginf(value):
            return "-inf"
        if value == 0:
            return "0"
        if value == f.max:
            return "max"
        if value == f.min:
            return "min"
        if value == f.tiny:
            return "tiny"
        if value == -f.tiny:
            return "-tiny"
        return f"{value:1.0e}".replace("e-0", "e-").replace("e+0", "e").replace("e+", "e").replace("e0", "")

    @classmethod
    def fromstring(cls, text):
        image = cls()
        row = 0
        col = 0
        for i, line in enumerate(text.splitlines()):
            image._allocate(row + i, col + len(line) - 1)
            image.table[row + i : row + i + 1, col : col + len(line)] = line
        return image

    @classmethod
    def fromseq(cls, seq, colstart="", colsep=" ", colend="", mintextwidth=None, maxtextwidth=None):
        image = cls()
        col_widths = defaultdict(int)
        data = []
        for i, row in enumerate(seq):
            d = []
            for j, value in enumerate(row):
                text = cls._value_to_text(value)
                col_widths[j] = max(col_widths[j], len(text))
                d.append(text)
            data.append(d)

        if maxtextwidth is None:
            maxtextwidth = max(col_widths.values())
        if mintextwidth is None:
            mintextwidth = 0

        offsets = [0]
        for j, w in sorted(col_widths.items()):
            offsets.append(
                offsets[-1] + (len(colstart) if j > 0 else 0) + min(max(w, mintextwidth), maxtextwidth) + len(colsep)
            )

        image._allocate(len(data) - 1, offsets[-1] + len(colend) - len(colsep))

        for i, row in enumerate(data):
            for j, text in enumerate(row):
                prefix = colstart if j == 0 else ""
                suffix = colsep if j < len(row) - 1 else colend
                t = text[:maxtextwidth]
                if len(t) < mintextwidth:
                    t += " " * (mintextwidth - len(t))
                image.insert(i, offsets[j], prefix + t + suffix)

        return image

    def tostring(self):
        lst = []
        for row in self.table:
            lst.append((b"".join(row)).decode().rstrip())
        return "\n".join(lst)

    def __str__(self):
        return self.tostring()

    def append(self, row, col, *args, **kwargs):
        row, col = self._handle_negative_location(row, col)
        self.insert(row + 1, col, *args, **kwargs)

    def insert(self, row, col, subimage, loc="ul"):
        """Insert subimage to image.

        Insertion will override existing cells of the current text image.

        Parameters
        ----------
        row, col: (int, int)
          The coordinates of loc within the subimage.
        subimage: {TextImage, ndarray, str}
          Subimage.
        loc: {'ul', 'uc', 'ur', 'l', 'c', 'r', 'll', 'lc', 'lr'}
          The location within subimage.
        """
        if isinstance(subimage, str):
            subimage = TextImage.fromstring(subimage)

        if isinstance(subimage, TextImage):
            subimage = subimage.table

        assert isinstance(subimage, numpy.ndarray), type(subimage)

        self.get_view(row, col, subimage.shape, loc=loc)[:] = subimage

    def get_view(self, row, col, shape, loc="ul"):
        """Return subimage."""
        row, col = self._handle_negative_location(row, col)
        row, col = self._shift_by_loc(row, col, loc, shape)
        self._allocate(row + shape[0] - 1, col + shape[1] - 1)
        return self.table[row : row + shape[0], col : col + shape[1]]

    def fill(self, row, col, mask, loc="ul", symbol="x"):
        """Fill subimage with symbol per mask."""
        assert mask.dtype == numpy.bool_, mask.dtype
        textmask = self.get_view(row, col, mask.shape, loc=loc)
        textmask[numpy.where(mask)] = symbol

        # textmask = numpy.full(mask.shape, self.bg, dtype='c')
        # textmask[numpy.where(mask)] = symbol
        # self.insert(row, col, textmask)
