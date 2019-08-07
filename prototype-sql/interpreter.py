# Simple, slow interpreter of row-wise data. Everything is a data.Instance.

import numpy

import data
import parser

################################################################################ utils

class SymbolTable:
    def __init__(self, parent=None):
        self._parent = parent
        self._table = {}

    def __repr__(self):
        return "<SymbolTable ({0} symbols) {1}>".format(len(self._table), repr(self._parent))

    def get(self, where, line=None, source=None):
        if where in self._table:
            return self._table[where]
        elif self._parent is not None:
            return self._parent[where]
        else:
            raise parser.LanguageError("unrecognized variable or function name: {0}".format(repr(where)), self._line, self._source)

    def __getitem__(self, where):
        return self.get(where)

    def __setitem__(self, where, what):
        self._table[where] = what

class Counter:
    def __init__(self, line=None, source=None):
        self._line, self._source = line, source
        self._table = {}
        self._n = 0
        self._sumw = 0.0
        self._sumw2 = 0.0

    @property
    def entries(self):
        return self._n

    @property
    def value(self):
        return self._sumw

    @property
    def error(self):
        return numpy.sqrt(self._sumw2)

    def __repr__(self):
        return "<Counter {0} ({1} +- {2})>".format(self.entries, self.value, self.error)

    def fill(self, w):
        self._n += 1
        self._sumw += w
        self._sumw2 += w**2

    def iterkeys(self, recursive=False):
        for n, x in self._table.items():
            yield n
            if recursive and isinstance(x, Counter):
                for n2 in x.iterkeys(recursive=recursive):
                    yield n + "/" + n2

    def keys(self, recursive=False):
        return list(self.iterkeys(recursive=recursive))

    def allkeys(self):
        return keys(recursive=True)

    def __contains__(self, where):
        return where in self._table

    def __getitem__(self, where):
        try:
            i = where.index("/")
        except ValueError:
            return self._table[where]
        else:
            return self._table[where[:i]][where[i + 1:]]

    def __setitem__(self, where, what):
        self._table[where] = what

class Binning: pass

class Unspecified(Binning):
    def __repr__(self):
        return "Unspecified()"

    def num(self, data):
        return max(10, 2*len(data)**(3**-1))

    def range(self, data):
        return None

class Regular(Binning):
    def __init__(self, num, low, high):
        self._num, self._low, self._high = num, low, high

    def __repr__(self):
        return "Regular({0}, {1}, {2})".format(self._num, self._low, self._high)

    def num(self, data):
        return self._num

    def range(self, data):
        return (self._low, self._high)

class Histogram(Counter):
    def __init__(self, binnings, line=None, source=None):
        self._binnings, self._line, self._source = binnings, line, source
        self._data = []
        self._weights = []

    def __repr__(self):
        return "<Histogram {0} dim {1} entries>".format(len(self._binnings), len(self._data))

    def fill(self, x, w):
        assert len(x) == len(self._binnings)
        self._data.append(x)
        self._weights.append([w])

    def numpy(self):
        if len(self._binnings) == 1:
            return numpy.histogram(self._data, bins=self._binnings[0].num(self._data), range=self._binnings[0].range(self._data), weights=self._weights)
        else:
            return numpy.histogramdd(self._data, bins=[x.num(self._data) for x in self._binnings], range=[x.range(self._data) for x in self._binnings], weights=self._weights)

################################################################################ functions

fcns = SymbolTable()







################################################################################ run

def runstep(node, symbols, counter, weight):
    if isinstance(node, parser.Literal):
        return node.value

    elif isinstance(node, parser.Symbol):
        return symbols[node.symbol]

    elif isinstance(node, parser.Block):
        raise NotImplementedError(node)

    elif isinstance(node, parser.Call):
        raise NotImplementedError(node)

    elif isinstance(node, parser.GetItem):
        raise NotImplementedError(node)

    elif isinstance(node, parser.GetAttr):
        raise NotImplementedError(node)

    elif isinstance(node, parser.Choose):
        raise NotImplementedError(node)

    elif isinstance(node, parser.TableBlock):
        raise NotImplementedError(node)

    elif isinstance(node, parser.GroupBy):
        raise NotImplementedError(node)

    elif isinstance(node, parser.MinMaxBy):
        raise NotImplementedError(node)

    elif isinstance(node, parser.Assignment):
        raise NotImplementedError(node)

    elif isinstance(node, parser.Histogram):
        if node.name not in counter:
            binnings = []
            for axis in node.axes:
                if axis.binning is None:
                    binnings.append(Unspecified())

                elif (isinstance(axis.binning, parser.Call) and
                      isinstance(axis.binning.function, parser.Symbol) and axis.binning.function.symbol == "regular" and
                      len(axis.binning.arguments) == 3 and
                      isinstance(axis.binning.arguments[0], parser.Literal) and isinstance(axis.binning.arguments[0].value, int) and
                      isinstance(axis.binning.arguments[1], parser.Literal) and isinstance(axis.binning.arguments[1].value, (int, float)) and
                      isinstance(axis.binning.arguments[2], parser.Literal) and isinstance(axis.binning.arguments[2].value, (int, float))):
                    binnings.append(Regular(axis.binning.arguments[0].value, axis.binning.arguments[1].value, axis.binning.arguments[2].value))

                else:
                    raise parser.LanguageError("histogram binning must match one of these patterns: regular(int, float, float)", node.line, node.source)

            counter[node.name] = Histogram(binnings, line=node.line, source=node.source)

        datum = []
        for axis in node.axes:
            component = runstep(axis.expression, symbols, counter, weight)
            if isinstance(component, data.Instance) and isinstance(component.value, (int, float)):
                datum.append(component.value)
            elif component is None:
                datum.append(None)
            else:
                raise parser.LanguageError("histograms can only be filled with numbers (not lists of numbers or records)", axis.line, axis.source)

        if all(x is not None for x in datum):
            counter[node.name].fill(datum, weight)
        elif any(x is not None for x in datum):
            raise parser.LanguageError("components must all be missing or none be missing", node.line, node.source)

    elif isinstance(node, parser.Axis):
        raise NotImplementedError(node)

    elif isinstance(node, parser.Vary):
        raise NotImplementedError(node)

    elif isinstance(node, parser.Trial):
        raise NotImplementedError(node)

    elif isinstance(node, parser.Cut):
        raise NotImplementedError(node)

def run(source, dataset):
    if not isinstance(source, parser.AST):
        source = parser.parse(source)

    counter = Counter()
    for entry in dataset:
        if not isinstance(entry, data.RecordInstance):
            raise parser.LanguageError("entries must be records (outermost array structure must be RecordArray)")

        symbols = SymbolTable(fcns)
        for n in entry.fields():
            symbols[n] = entry[n]

        counter.fill(1.0)
        for node in source:
            runstep(node, symbols, counter, 1.0)

    return counter

################################################################################ tests

def test_dataset():
    events = data.RecordArray({
        "muons": data.ListArray([0, 3, 3, 5], [3, 3, 5, 9], data.RecordArray({
            "pt": data.PrimitiveArray([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]),
            "iso": data.PrimitiveArray([0, 0, 100, 50, 30, 1, 2, 3, 4])
        })),
        "jets": data.ListArray([0, 5, 6, 8], [5, 6, 8, 12], data.RecordArray({
            "pt": data.PrimitiveArray([1, 2, 3, 4, 5, 100, 30, 50, 1, 2, 3, 4]),
            "mass": data.PrimitiveArray([10, 10, 10, 10, 10, 5, 15, 15, 9, 8, 7, 6])
        })),
        "met": data.PrimitiveArray([100, 200, 300, 400])
    })
    events.setindex()
    return data.instantiate(events)

def test_hist():
    counter = run(r"""
hist met
""", test_dataset())
    assert (counter.entries, counter.value, counter.error) == (4, 4.0, 2.0)
    assert counter.keys() == ["0"]
    counts, edges = counter["0"].numpy()
    assert counts.tolist() == [1, 0, 0, 1, 0, 0, 1, 0, 0, 1]
    assert edges.tolist() == [100, 130, 160, 190, 220, 250, 280, 310, 340, 370, 400]
