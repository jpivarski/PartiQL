# Database-style surrogate keys.

import functools

class Ref:
    @staticmethod
    def new():
        out = Ref(Ref.numrefs)
        Ref.numrefs += 1
        return out

    numrefs = 0

    def __init__(self, num):
        self.num = num

    def __repr__(self):
        return "<Ref {0}>".format(self.num)

    def __str__(self):
        return "#" + str(self.num)

    def __eq__(self, other):
        return type(other) is Ref and self.num == other.num

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((Ref, self.num))

class CrossRef(Ref):
    @staticmethod
    def fromtuple(tup):
        if not isinstance(tup, tuple):
            tup = tuple(tup)
        assert len(tup) != 0
        if len(tup) == 1:
            return tup[0]
        else:
            return CrossRef(tup[0], CrossRef.fromtuple(tup[1:]))

    def __init__(self, left, right):
        self.left, self.right = left, right

    def __repr__(self):
        return "CrossRef({0}, {1})".format(repr(self.left), repr(self.right))

    def __str__(self):
        return "{0}*{1}".format(str(self.left), str(self.right))

    def __eq__(self, other):
        return type(other) is CrossRef and self.left == other.left and self.right == other.right

    def __hash__(self):
        return hash((CrossRef, self.left, self.right))

class RowIndex:
    "RowIndex is array-like, with each entry specifying the sequence of integer indexes as a path from root to tree-node. Referential identity is important, as it determines which arrays are compatible in ufunc-like operations."

    def __init__(self, array, ref=None):
        "In this prototype, 'array' is a list of equal-length tuples."
        self.array = array
        self.ref = Ref.new() if ref is None else ref

    def __repr__(self):
        return "RowIndex({0}, {1})".format(repr(self.array), repr(self.ref))

    def __str__(self):
        return "{0}({1})".format(str(self.ref), " ".join(repr(x) for x in self.array))

    def tolist(self):
        return list(self.array)

    def __eq__(self, other):
        if isinstance(other, RowIndex):
            return self.ref == other.ref and self.tolist() == other.tolist()
        else:
            return self.tolist() == other

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((RowIndex, self.array, self.ref))

    def same(self, other):
        return isinstance(other, RowIndex) and self.ref == other.ref

    def __iter__(self):
        for x in self.array:
            yield x

    def __getitem__(self, where):
        if isinstance(where, int):
            return RowKey(self.array[where], self.ref)
        elif isinstance(where, slice):
            return RowIndex(self.array[where], self.ref)
        else:
            raise NotImplementedError(where)

class ColIndex:
    "ColIndex is AST-like, specifying the sequence of string indexes as a path from root to tree-node or join operations performed to build the object. Referential identity is not important; ColIndexes should be compared by value."

    def __init__(self, *path):
        "In this prototype, 'path' is a tuple of strings."
        self.path = path

    def __repr__(self):
        return "ColIndex({0})".format(", ".join(repr(x) for x in self.path))

    def __str__(self):
        return "({0})".format(" ".join(repr(x) for x in self.path))

    def tolist(self):
        return list(self.path)

    def __eq__(self, other):
        if isinstance(other, ColIndex):
            return self.path == other.path
        else:
            return self.path == other

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((ColIndex, self.path))

    def __getitem__(self, where):
        if isinstance(where, int):
            return ColKey((self.path[where],))
        elif isinstance(where, slice):
            return ColIndex(*self.path[where])
        else:
            raise NotImplementedError(where)

    def key(self):
        return ColKey(self.path)

    def withattr(self, attr):
        return ColIndex(*(self.path + (attr,)))

@functools.total_ordering
class RowKey:
    "RowKey is an element of a RowIndex, representing a unique row by reference."

    def __init__(self, index, ref):
        self.index, self.ref = index, ref

    def __repr__(self):
        return "RowKey({0}, {1})".format(repr(self.index), repr(self.ref))

    def __str__(self):
        return "{0}({1})".format(str(self.ref), " ".join(repr(x) for x in self.index))

    def __eq__(self, other):
        return isinstance(other, RowKey) and self.index == other.index and self.ref == other.ref

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((RowKey, self.index, self.ref))

    def __lt__(self, other):
        if not isinstance(other, RowKey):
            raise TypeError("inequalities not supported between instances of 'RowKey' and '{0}'".format(type(other)))
        if self.ref != other.ref:
            raise TypeError("inequalities not supported between RowKeys with different references: {0} and {1}".format(str(self.ref), str(other.ref)))
        return self.index < other.index

class ColKey:
    "ColKey is an element of a ColIndex, representing a unique column by value."

    def __init__(self, index):
        self.index = index

    def __repr__(self):
        return "ColKey({0})".format(repr(self.index))

    def __str__(self):
        return "({0})".format(" ".join(repr(x) for x in self.index))

    def __eq__(self, other):
        return type(self) is type(other) and self.index == other.index

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((ColKey, self.index))

class DerivedColKey(ColKey):
    def __init__(self, node):
        self.node = node

    def __repr__(self):
        return "DerivedColKey({0})".format(repr(self.node))

    def __str__(self):
        return "({0})".format(str(self.node))

    def __eq__(self, other):
        return type(self) is type(other) and self.node == other.node

    def __hash__(self):
        return hash((DerivedKey, self.node))
