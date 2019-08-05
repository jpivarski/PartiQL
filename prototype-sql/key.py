# Database-style surrogate keys.

class RowKey:
    "RowKey is array-like, with each entry specifying the sequence of integer indexes as a path from root to tree-node. Referential identity is important, as it determines which arrays are compatible in ufunc-like operations."

    def __init__(self, array):
        "In this prototype, 'array' is a list of equal-length tuples."
        self._array = array

    def __repr__(self):
        return "<RowKey {0} at 0x{1:012x}>".format(self.tolist(), id(self))

    def __str__(self):
        return str(self.tolist())

    def tolist(self):
        return list(self._array)

    def __eq__(self, other):
        if isinstance(other, RowKey):
            return self.tolist() == other.tolist()
        else:
            return self.tolist() == other

    def __ne__(self, other):
        return not self.__eq__(other)

    def __iter__(self):
        for x in self._array:
            yield x

class ColKey:
    "ColKey is AST-like, specifying the sequence of string indexes as a path from root to tree-node or join operations performed to build the object. Referential identity is not important; ColKeys should be compared by value."

    def __init__(self, *path):
        "In this prototype, 'path' is a tuple of strings."
        self._path = path

    def __repr__(self):
        return "ColKey({0})".format(", ".join(repr(x) for x in self._path))

    def tolist(self):
        return list(self._path)

    def __eq__(self, other):
        return isinstance(other, ColKey) and self._path == other._path

    def __ne__(self, other):
        return not self.__eq__(other)

    def withattr(self, attr):
        return ColKey(*(self._path + (attr,)))
