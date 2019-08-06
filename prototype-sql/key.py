# Database-style surrogate keys.

class RowKey:
    "RowKey is array-like, with each entry specifying the sequence of integer indexes as a path from root to tree-node. Referential identity is important, as it determines which arrays are compatible in ufunc-like operations."

    @staticmethod
    def newid():
        out = RowKey.numids
        RowKey.numids += 1
        return out

    numids = 0

    def __init__(self, array, id=None):
        "In this prototype, 'array' is a list of equal-length tuples."
        self._array = array
        self._id = RowKey.newid() if id is None else id

    def __repr__(self):
        return "<RowKey {0}: {1}>".format(self._id, self.tolist())

    def __str__(self):
        return str(self.tolist())

    def tolist(self):
        return list(self._array)

    def __eq__(self, other):
        if isinstance(other, RowKey):
            return self._id == other._id and self.tolist() == other.tolist()
        else:
            return self.tolist() == other

    def __ne__(self, other):
        return not self.__eq__(other)

    def same(self, other):
        return isinstance(other, RowKey) and self._id == other._id

    def __iter__(self):
        for x in self._array:
            yield x

    def __getitem__(self, where):
        if isinstance(where, int):
            return Item(self._id, self._array[where])
        elif isinstance(where, slice):
            return RowKey(self._array[where], self._id)
        else:
            raise NotImplementedError(where)

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
        if isinstance(other, ColKey):
            return self._path == other._path
        else:
            return self._path == other

    def __ne__(self, other):
        return not self.__eq__(other)

    def __getitem__(self, where):
        if isinstance(where, int):
            return Item(self._path[where])
        elif isinstance(where, slice):
            return ColKey(*self._path[where])
        else:
            raise NotImplementedError(where)

    def withattr(self, attr):
        return ColKey(*(self._path + (attr,)))

class Item:
    "Item is an element of a RowKey or a ColKey, representing a unique object by reference."

    def __init__(self, id, key):
        self._id, self._key = id, key

    def __repr__(self):
        return "<Item {0}: {1}>".format(self._id, self._key)

    def __eq__(self, other):
        return isinstance(other, Item) and self._id == other._id and self._key == other._key

    def __ne__(self, other):
        return not self.__eq__(other)
