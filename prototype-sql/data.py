# This is a simplified implementation of awkward classes to demonstrate a prototype SQL.
#
# PLUR types: Primitive, List, Union, Record

import index

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

    def setindex(self, row=None, col=None):
        self.row = index.RowIndex([(i,) for i in range(len(self))]) if row is None else row
        self.col = index.ColIndex() if col is None else col

class PrimitiveArray(Array):
    "Array of fixed-bytewidth objects: booleans, numbers, etc."

    def __init__(self, data, row=None, col=None):
        self.data, self.row, self.col = data, row, col

    def __getitem__(self, where):
        if isinstance(where, slice):
            return PrimitiveArray(self.data[where], None if self.row is None else self.row[where], self.col)
        elif isinstance(where, int):
            return self.data[where]
        else:
            raise TypeError("PrimitiveArray cannot be selected with {0}".format(where))

    def __len__(self):
        return len(self.data)

    def tolist(self):
        return list(self.data)

class ListArray(Array):
    "Array of variable-length but single-type lists (a.k.a. JaggedArray)."

    def __init__(self, starts, stops, content, row=None, col=None):
        self.starts, self.stops, self.content, self.row, self.col = starts, stops, content, row, col

    def __getitem__(self, where):
        if isinstance(where, str):
            return ListArray(self.starts, self.stops, self.content[where], self.row, None if self.col is None else self.col[1:])
        elif isinstance(where, slice):
            return ListArray(self.starts[where], self.stops[where], self.content, None if self.row is None else self.row[where], self.col)
        elif isinstance(where, int):
            return self.content[self.starts[where]:self.stops[where]]
        else:
            raise TypeError("ListArray cannot be selected with {0}".format(where))

    def __len__(self):
        return len(self.starts)

    def setindex(self, row=None, col=None):
        super(ListArray, self).setindex(row, col)
        subrow = [None] * len(self.content)
        for i, r in enumerate(self.row):
            for j in range(self.stops[i] - self.starts[i]):
                subrow[self.starts[i] + j] = r + (j,)
        self.content.setindex(index.RowIndex(subrow), col)

class UnionArray(Array):
    "Array of possibly multiple types (a.k.a. tagged union/sum type)."

    def __init__(self, tags, offsets, contents, row=None, col=None):
        self.tags, self.offsets, self.contents, self.row, self.col = tags, offsets, contents, row, col

    def __getitem__(self, where):
        if isinstance(where, str):
            return UnionArray(self.tags, self.offsets, [x[where] for x in self.contents], self.row, None if self.col is None else self.col[1:])
        elif isinstance(where, slice):
            return UnionArray(self.tags[where], self.offsets[where], self.contents, None if self.row is None else self.row[where], self.col)
        elif isinstance(where, int):
            return self.contents[self.tags[where]][self.offsets[where]]
        else:
            raise TypeError("UnionArray cannot be selected with {0}".format(where))

    def __len__(self):
        return len(self.tags)

    def setindex(self, row=None, col=None):
        super(UnionArray, self).setindex(row, col)
        subrows = [[None] * len(x) for x in self.contents]
        for i, r in enumerate(self.row):
            subrows[self.tags[i]][self.offsets[i]] = r
        for subrow, x in zip(subrows, self.contents):
            x.setindex(index.RowIndex(subrow), col)

class RecordArray(Array):
    "Array of record objects (a.k.a. Table, array of structs/product type)."

    def __init__(self, contents, row=None, col=None):
        self.contents, self.row, self.col = contents, row, col

    def __getitem__(self, where):
        if isinstance(where, str):
            return self.contents[where]
        elif isinstance(where, slice):
            return RecordArray({n: x[where] for n, x in self.contents.items()}, None if self.row is None else self.row[where], self.col)
        elif isinstance(where, int):
            return {n: x[where] for n, x in self.contents.items()}
        else:
            raise TypeError("RecordArray cannot be selected with {0}".format(where))

    def __len__(self):
        return min(len(x) for x in self.contents.values())

    def tolist(self):
        contents = {n: x.tolist() for n, x in self.contents.items()}
        return [{n: x[i] for n, x in contents.items()} for i in range(len(self))]

    def setindex(self, row=None, col=None):
        super(RecordArray, self).setindex(row, col)
        for n, x in self.contents.items():
            x.setindex(self.row, index.ColIndex(n) if col is None else col.withattr(n))

class Instance:
    def __init__(self, value, row, col):
        self.value, self.row, self.col = value, row, col

    def same(self, other):
        return type(self) is type(other) and (self.row == other.row and self.col == other.col)

    def __eq__(self, other):
        return type(self) is type(other) and (self.same(other) or self.value == other.value)

    def __ne__(self, other):
        return not self.__eq__(other)

class ValueInstance(Instance):
    name = "Value"

    def __repr__(self, indent=""):
        return indent + "{0}{1}{{ {2} }}".format(self.name, "" if self.row is None else str(self.row), repr(self.value))

    def tolist(self):
        return self.value

class ListInstance(Instance):
    name = "List"

    def __repr__(self, indent=""):
        out = [indent, "{0}{1}".format(self.name, "" if self.row is None else str(self.row)), "{ \n"]
        for x in self.value:
            out.append(x.__repr__(indent + "    ") + "\n")
        out.append(indent + "}")
        return "".join(out)

    def __getitem__(self, where):
        return self.value[where]

    def __iter__(self):
        for x in self.value:
            yield x

    def newempty(self):
        return ListInstance([], self.row, self.col)

    def append(self, x):
        self.value.append(x)

    def tolist(self):
        return [x.tolist() for x in self.value]

class RecordInstance(Instance):
    name = "Rec"

    def fields(self):
        return self.value.keys()

    def __repr__(self, indent=""):
        out = [indent, "{0}{1}".format(self.name, "" if self.row is None else str(self.row)), "{ \n"]
        for n, x in self.value.items():
            out.append(indent + "    " + n + " = " + x.__repr__(indent + "    ").lstrip(" ") + "\n")
        out.append(indent + "}")
        return "".join(out)

    def __contains__(self, where):
        return where in self.value

    def __getitem__(self, where):
        return self.value[where]

    def __setitem__(self, where, what):
        self.value[where] = what

    def newempty(self):
        return RecordInstance({}, self.row, self.col)

    def tolist(self):
        return {n: x.tolist() for n, x in self.value.items()}

def instantiate(data):
    def recurse(array, i):
        if isinstance(array, PrimitiveArray):
            return ValueInstance(array.data[i], array.row[i], array.col.key())
        elif isinstance(array, ListArray):
            return ListInstance([recurse(array.content, j) for j in range(array.starts[i], array.stops[i])], array.row[i], array.col.key())
        elif isinstance(array, UnionArray):
            return recurse(array.contents[array.tags[i]], array.offsets[i])
        elif isinstance(array, RecordArray):
            return RecordInstance({n: recurse(x, i) for n, x in array.contents.items()}, array.row[i], array.col.key())
        else:
            raise NotImplementedError
    return ListInstance([recurse(data, i) for i in range(len(data))], None, None)

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

    events.setindex()

    muonpt = events.contents["muons"].content.contents["pt"]
    assert muonpt.row == [(0, 0), (0, 1), (0, 2), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2), (3, 3)]
    assert muonpt.col == ("muons", "pt")

    muoniso = events.contents["muons"].content.contents["iso"]
    assert muonpt.row == muoniso.row
    assert muonpt.row.same(muoniso.row)

    c1, c2 = muonpt.col.tolist()
    for i, (r1, r2) in enumerate(muonpt.row):
        assert events[c1][c2][r1][r2] == muonpt[i]

    instantiate(events)

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

    egamma.setindex()
    assert egamma.contents[0].contents["pt"].row == [(0,), (1,), (3,), (7,), (8,)]
    assert egamma.contents[1].contents["pt"].row == [(2,), (4,), (5,), (6,)]
    assert egamma.contents[0].contents["pt"].col == ("pt",)
    assert egamma.contents[1].contents["pt"].col == ("pt",)

    instantiate(egamma)
