# This is a simplified implementation of awkward classes to demonstrate a prototype SQL.
#
# PLUR types: Primitive, List, Union, Record

import key

class Array:
    "Abstract base class."

    def __init__(self):
        raise TypeError("cannot instantiate an abstract base class")

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __repr__(self):
        return "<Array {0}>".format(str(self))

    def __str__(self):
        return str(self.tolist())

    def tolist(self):
        return [x.tolist() if isinstance(x, Array) else x for x in self]

    def __eq__(self, other):
        if isinstance(other, Array):
            return self.tolist() == other.tolist()
        else:
            return self.tolist() == other

    def __ne__(self, other):
        return not self.__eq__(other)

    def setkey(self, row=None, col=None):
        self._row = key.RowKey([(i,) for i in range(len(self))]) if row is None else row
        self._col = key.ColKey() if col is None else col

class PrimitiveArray(Array):
    "Array of fixed-bytewidth objects: booleans, numbers, etc."

    def __init__(self, data, row=None, col=None):
        self._data, self._row, self._col = data, row, col

    def __getitem__(self, where):
        if isinstance(where, slice):
            return PrimitiveArray(self._data[where], None if self._row is None else self._row[where], self._col)
        elif isinstance(where, int):
            return self._data[where]
        else:
            raise TypeError("PrimitiveArray cannot be selected with {0}".format(where))

    def __len__(self):
        return len(self._data)

    def tolist(self):
        return list(self._data)

class ListArray(Array):
    "Array of variable-length but single-type lists (a.k.a. JaggedArray)."

    def __init__(self, starts, stops, content, row=None, col=None):
        self._starts, self._stops, self._content, self._row, self._col = starts, stops, content, row, col

    def __getitem__(self, where):
        if isinstance(where, str):
            return ListArray(self._starts, self._stops, self._content[where], self._row, None if self._col is None else self._col[1:])
        elif isinstance(where, slice):
            return ListArray(self._starts[where], self._stops[where], self._content, None if self._row is None else self._row[where], self._col)
        elif isinstance(where, int):
            return self._content[self._starts[where]:self._stops[where]]
        else:
            raise TypeError("ListArray cannot be selected with {0}".format(where))

    def __len__(self):
        return len(self._starts)

    def setkey(self, row=None, col=None):
        super(ListArray, self).setkey(row, col)
        subrow = [None] * len(self._content)
        for i, r in enumerate(self._row):
            for j in range(self._stops[i] - self._starts[i]):
                subrow[self._starts[i] + j] = r + (j,)
        self._content.setkey(key.RowKey(subrow), col)

class UnionArray(Array):
    "Array of possibly multiple types (a.k.a. tagged union/sum type)."

    def __init__(self, tags, index, contents, row=None, col=None):
        self._tags, self._index, self._contents, self._row, self._col = tags, index, contents, row, col

    def __getitem__(self, where):
        if isinstance(where, str):
            return UnionArray(self._tags, self._index, [x[where] for x in self._contents], self._row, None if self._col is None else self._col[1:])
        elif isinstance(where, slice):
            return UnionArray(self._tags[where], self._index[where], self._contents, None if self._row is None else self._row[where], self._col)
        elif isinstance(where, int):
            return self._contents[self._tags[where]][self._index[where]]
        else:
            raise TypeError("UnionArray cannot be selected with {0}".format(where))

    def __len__(self):
        return len(self._tags)

    def setkey(self, row=None, col=None):
        super(UnionArray, self).setkey(row, col)
        subrows = [[None] * len(x) for x in self._contents]
        for i, r in enumerate(self._row):
            subrows[self._tags[i]][self._index[i]] = r
        for subrow, x in zip(subrows, self._contents):
            x.setkey(key.RowKey(subrow), col)

class RecordArray(Array):
    "Array of record objects (a.k.a. Table, array of structs/product type)."

    def __init__(self, contents, row=None, col=None):
        self._contents, self._row, self._col = contents, row, col

    def __getitem__(self, where):
        if isinstance(where, str):
            return self._contents[where]
        elif isinstance(where, slice):
            return RecordArray({n: x[where] for n, x in self._contents.items()}, None if self._row is None else self._row[where], self._col)
        elif isinstance(where, int):
            return {n: x[where] for n, x in self._contents.items()}
        else:
            raise TypeError("RecordArray cannot be selected with {0}".format(where))

    def __len__(self):
        return min(len(x) for x in self._contents.values())

    def tolist(self):
        contents = {n: x.tolist() for n, x in self._contents.items()}
        return [{n: x[i] for n, x in contents.items()} for i in range(len(self))]

    def setkey(self, row=None, col=None):
        super(RecordArray, self).setkey(row, col)
        for n, x in self._contents.items():
            x.setkey(self._row, key.ColKey(n) if col is None else col.withattr(n))

class Instance:
    def __init__(self, value, row, col):
        self.value, self.row, self.col = value, row, col

    def __repr__(self):
        return "<{0} at {1}, {2}: {3}>".format(type(self).__name__, self.row, self.col, self.value)

    def same(self, other):
        return isinstance(other, Instance) and (self.row == other.row and self.col == other.col)

    def __eq__(self, other):
        return isinstance(other, Instance) and (self.same(other) or self.value == other.value)

    def __ne__(self, other):
        return not self.__eq__(other)

class ListInstance(Instance):
    def __getitem__(self, where):
        return self.value[where]

    def __iter__(self):
        for x in self.value:
            yield x

class RecordInstance(Instance):
    def __getitem__(self, where):
        return self.value[where]

def instantiate(data):
    def recurse(array, start, stop):
        if isinstance(array, PrimitiveArray):
            if stop is None:
                return Instance(array._data[start], array._row[start], array._col)
            else:
                return ListInstance([recurse(array, i, None) for i in range(start, stop)], array._row[start:stop], array._col)

        elif isinstance(array, ListArray):
            if stop is None:
                return recurse(array._content, array._starts[start], array._stops[start])
            else:
                return ListInstance([recurse(array._content, array._starts[i], array._stops[i]) for i in range(start, stop)], array._row[start:stop], array._col)

        elif isinstance(array, UnionArray):
            raise NotImplementedError

        elif isinstance(array, RecordArray):
            if stop is None:
                return RecordInstance({n: recurse(x, start, stop) for n, x in array._contents.items()}, array._row[start], array._col)
            else:
                return ListInstance([recurse(array, i, None) for i in range(start, stop)], array._row[start:stop], array._col)

    return ListInstance([recurse(data, i, None) for i in range(len(data))], None, None)

################################################################################ tests

def test_data():
    # data in columnar form
    events = RecordArray({
        "muons": ListArray([0, 3, 3, 5], [3, 3, 5, 9], RecordArray({
            "pt": PrimitiveArray([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]),
            "iso": PrimitiveArray([0, 0, 100, 50, 30, 1, 2, 3, 4])
        })),
        "jets": ListArray([0, 5, 6, 8], [5, 6, 8, 12], RecordArray({
            "pt": PrimitiveArray([1, 2, 3, 4, 5, 100, 30, 50, 1, 2, 3, 4]),
            "mass": PrimitiveArray([10, 10, 10, 10, 10, 5, 15, 15, 9, 8, 7, 6])
        })),
        "met": PrimitiveArray([100, 200, 300, 400])
    })

    # same data in rowwise form
    assert events == [
        {'muons': [
            {'pt': 1.1, 'iso': 0},
            {'pt': 2.2, 'iso': 0},
            {'pt': 3.3, 'iso': 100}],
         'jets': [
            {'pt': 1, 'mass': 10},
            {'pt': 2, 'mass': 10},
            {'pt': 3, 'mass': 10},
            {'pt': 4, 'mass': 10},
            {'pt': 5, 'mass': 10}],
         'met': 100},
        {'muons': [],
         'jets': [{'pt': 100, 'mass': 5}],
         'met': 200},
        {'muons': [
            {'pt': 4.4, 'iso': 50},
            {'pt': 5.5, 'iso': 30}],
         'jets': [
            {'pt': 30, 'mass': 15},
            {'pt': 50, 'mass': 15}],
         'met': 300},
        {'muons': [
            {'pt': 6.6, 'iso': 1},
            {'pt': 7.7, 'iso': 2},
            {'pt': 8.8, 'iso': 3},
            {'pt': 9.9, 'iso': 4}],
         'jets': [
            {'pt': 1, 'mass': 9},
            {'pt': 2, 'mass': 8},
            {'pt': 3, 'mass': 7},
            {'pt': 4, 'mass': 6}],
         'met': 400}]

    # projection down to the numerical values
    assert events["muons"]["pt"] == [[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6, 7.7, 8.8, 9.9]]

    # single record object
    assert events[0] == {
        'muons': [
            {'pt': 1.1, 'iso': 0},
            {'pt': 2.2, 'iso': 0},
            {'pt': 3.3, 'iso': 100}],
        'jets': [
            {'pt': 1, 'mass': 10},
            {'pt': 2, 'mass': 10},
            {'pt': 3, 'mass': 10},
            {'pt': 4, 'mass': 10},
            {'pt': 5, 'mass': 10}],
        'met': 100}

    # integer and string indexes commute, but string-string and integer-integer do not
    assert events["muons"][0]          == events[0]["muons"]
    assert events["muons"][0]["pt"]    == events[0]["muons"]["pt"]
    assert events["muons"][0][2]       == events[0]["muons"][2]
    assert events["muons"][0]["pt"][2] == events[0]["muons"]["pt"][2]
    assert events["muons"][0]["pt"][2] == events[0]["muons"][2]["pt"]
    assert events["muons"][0]["pt"][2] == events["muons"][0][2]["pt"]
    assert events["muons"][0]["pt"][2] == events["muons"]["pt"][0][2]

    events.setkey()

    muonpt = events._contents["muons"]._content._contents["pt"]
    assert muonpt._row == [(0, 0), (0, 1), (0, 2), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2), (3, 3)]
    assert muonpt._col == ("muons", "pt")

    muoniso = events._contents["muons"]._content._contents["iso"]
    assert muonpt._row == muoniso._row
    assert muonpt._row.same(muoniso._row)

    c1, c2 = muonpt._col.tolist()
    for i, (r1, r2) in enumerate(muonpt._row):
        assert events[c1][c2][r1][r2] == muonpt[i]

    egamma = UnionArray([0, 0, 1, 0, 1, 1, 1, 0, 0], [0, 1, 0, 2, 1, 2, 3, 3, 4], [
        RecordArray({
            "q": PrimitiveArray([1, -1, -1, 1, 1]),
            "pt": PrimitiveArray([10, 20, 30, 40, 50])
        }),
        RecordArray({
            "pt": PrimitiveArray([1.1, 2.2, 3.3, 4.4])
        })
    ])

    assert egamma == [
        {'pt': 10, 'q': 1},
        {'pt': 20, 'q': -1},
        {'pt': 1.1},
        {'pt': 30, 'q': -1},
        {'pt': 2.2},
        {'pt': 3.3},
        {'pt': 4.4},
        {'pt': 40, 'q': 1},
        {'pt': 50, 'q': 1}]

    assert egamma["pt"] == [10, 20, 1.1, 30, 2.2, 3.3, 4.4, 40, 50]

    egamma.setkey()
    assert egamma._contents[0]._contents["pt"]._row == [(0,), (1,), (3,), (7,), (8,)]
    assert egamma._contents[1]._contents["pt"]._row == [(2,), (4,), (5,), (6,)]
    assert egamma._contents[0]._contents["pt"]._col == ("pt",)
    assert egamma._contents[1]._contents["pt"]._col == ("pt",)
