# Database-style surrogate keys.

class RowIndex:
    "RowIndex is array-like, with each entry specifying the sequence of integer indexes as a path from root to tree-node. Referential identity is important, as it determines which arrays are compatible in ufunc-like operations."

    @staticmethod
    def newid():
        out = RowIndex.numids
        RowIndex.numids += 1
        return out

    numids = 0

    def __init__(self, array, id=None):
        "In this prototype, 'array' is a list of equal-length tuples."
        self._array = array
        self._id = RowIndex.newid() if id is None else id

    def __repr__(self):
        return "RowIndex({0}, {1})".format(repr(self._array), self._id)

    def __str__(self):
        return "#{0}({1})".format(self._id, " ".join(repr(x) for x in self._array))

    def tolist(self):
        return list(self._array)

    def __eq__(self, other):
        if isinstance(other, RowIndex):
            return self._id == other._id and self.tolist() == other.tolist()
        else:
            return self.tolist() == other

    def __ne__(self, other):
        return not self.__eq__(other)

    def same(self, other):
        return isinstance(other, RowIndex) and self._id == other._id

    def __iter__(self):
        for x in self._array:
            yield x

    def __getitem__(self, where):
        if isinstance(where, int):
            return RowKey(self._array[where], self._id)
        elif isinstance(where, slice):
            return RowIndex(self._array[where], self._id)
        else:
            raise NotImplementedError(where)

class ColIndex:
    "ColIndex is AST-like, specifying the sequence of string indexes as a path from root to tree-node or join operations performed to build the object. Referential identity is not important; ColIndexes should be compared by value."

    def __init__(self, *path):
        "In this prototype, 'path' is a tuple of strings."
        self._path = path

    def __repr__(self):
        return "ColIndex({0})".format(", ".join(repr(x) for x in self._path))

    def __str__(self):
        return "({0})".format(" ".join(repr(x) for x in self._path))

    def tolist(self):
        return list(self._path)

    def __eq__(self, other):
        if isinstance(other, ColIndex):
            return self._path == other._path
        else:
            return self._path == other

    def __ne__(self, other):
        return not self.__eq__(other)

    def __getitem__(self, where):
        if isinstance(where, int):
            return ColKey((self._path[where],))
        elif isinstance(where, slice):
            return ColIndex(*self._path[where])
        else:
            raise NotImplementedError(where)

    def key(self):
        return ColKey(self._path)

    def withattr(self, attr):
        return ColIndex(*(self._path + (attr,)))

class RowKey:
    "RowKey is an element of a RowIndex, representing a unique row by reference."

    def __init__(self, index, id):
        self._index, self._id = index, id

    def __repr__(self):
        return "RowKey({0}, {1})".format(repr(self._index), self._id)

    def __str__(self):
        return "#{0}({1})".format(self._id, " ".join(repr(x) for x in self._index))

    def __eq__(self, other):
        return isinstance(other, RowKey) and self._index == other._index and self._id == other._id

    def __ne__(self, other):
        return not self.__eq__(other)

class ColKey:
    "ColKey is an element of a ColIndex, representing a unique column by value."

    def __init__(self, index):
        self._index = index

    def __repr__(self):
        return "ColKey({0})".format(repr(self._index))

    def __str__(self):
        return "({0})".format(" ".join(repr(x) for x in self._index))

    def __eq__(self, other):
        return isinstance(other, ColKey) and self._index == other._index

    def __ne__(self, other):
        return not self.__eq__(other)