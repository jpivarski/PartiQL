# This is a simplified implementation of awkward classes to demonstrate a prototype SQL.
#
# PLUR types: Primitive, List, Union, Record

import key

class Array:
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
    def __init__(self, data):
        self._data = data

    def __getitem__(self, where):
        if isinstance(where, slice):
            return PrimitiveArray(self._data[where])
        elif isinstance(where, int):
            return self._data[where]
        else:
            raise TypeError("PrimitiveArray cannot be selected with {0}".format(where))

    def __len__(self):
        return len(self._data)

    def tolist(self):
        return list(self._data)

class ListArray(Array):
    def __init__(self, starts, stops, content):
        self._starts, self._stops, self._content = starts, stops, content

    def __getitem__(self, where):
        if isinstance(where, str):
            return ListArray(self._starts, self._stops, self._content[where])
        elif isinstance(where, slice):
            return ListArray(self._starts[where], self._stops[where], self._content)
        elif isinstance(where, int):
            return self._content[self._starts[where]:self._stops[where]]
        else:
            raise TypeError("ListArray cannot be selected with {0}".format(where))

    def __len__(self):
        return len(self._starts)

    def setkey(self, row=None, col=None):
        super(ListArray, self).setkey(row, col)
        subrow = [-1] * len(self._content)
        for i, r in enumerate(self._row):
            for j in range(self._stops[i] - self._starts[i]):
                subrow[self._starts[i] + j] = r + (j,)
        self._content.setkey(key.RowKey(subrow), col)

class UnionArray(Array):
    def __init__(self, tags, index, contents):
        self._tags, self._index, self._contents = tags, index, contents

    def __getitem__(self, where):
        if isinstance(where, str):
            return UnionArray(self._tags, self._index, [x[where] for x in self._contents])
        elif isinstance(where, slice):
            return UnionArray(self._tags[where], self._index[where], self._contents)
        elif isinstance(where, int):
            return self._contents[self._tags[where]][self._index[where]]
        else:
            raise TypeError("UnionArray cannot be selected with {0}".format(where))

    def __len__(self):
        return len(self._tags)

    def setkey(self, row=None, col=None):
        super(UnionArray, self).setkey(row, col)
        subrows = [[-1] * len(x) for x in self._contents]
        for i, r in enumerate(self._row):
            subrows[self._tags[i]][self._index[i]] = r
        for subrow, x in zip(subrows, self._contents):
            x.setkey(key.RowKey(subrow), col)

class RecordArray(Array):
    def __init__(self, contents):
        self._contents = contents

    def __getitem__(self, where):
        if isinstance(where, str):
            return self._contents[where]
        elif isinstance(where, slice):
            return RecordArray({n: x[where] for n, x in self._contents.items()})
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

################################################################################

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
    assert muonpt._row == key.RowKey([(0, 0), (0, 1), (0, 2), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2), (3, 3)])
    assert muonpt._col == key.ColKey("muons", "pt")

    muoniso = events._contents["muons"]._content._contents["iso"]
    assert muonpt._row == muoniso._row
    assert muonpt._row is muoniso._row

    c1, c2 = muonpt._col.tolist()
    for i, (r1, r2) in enumerate(muonpt._row):
        assert events[c1][c2][r1][r2] == muonpt[i]
