# Simple, slow interpreter of row-wise data. Everything is a data.Instance.

import math

import numpy

import index
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
            raise parser.LanguageError("unrecognized variable or function name: {0}".format(repr(where)), line, source)

    def __getitem__(self, where):
        return self.get(where)

    def __setitem__(self, where, what):
        self._table[where] = what

    def empty(self):
        return len(self._table) == 0

    def __iter__(self):
        for n in self._table:
            yield n

class Counter:
    def __init__(self, line=None, source=None):
        self._line, self._source = line, source
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
        return "<{0} {1} ({2} +- {3})>".format(type(self).__name__, self.entries, self.value, self.error)

    def fill(self, w):
        self._n += 1
        self._sumw += w
        self._sumw2 += w**2

class DirectoryCounter(Counter):
    def __init__(self, line=None, source=None):
        super(DirectoryCounter, self).__init__(line=line, source=source)
        self._n = 0
        self._sumw = 0.0
        self._sumw2 = 0.0
        self._table = {}

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
        super(Histogram, self).__init__(line=line, source=source)
        self._binnings = binnings
        self._data = []
        self._weights = []

    def __repr__(self):
        return "<Histogram {0} dim {1} entries>".format(len(self._binnings), len(self._data))

    def fill(self, x, w):
        assert len(x) == len(self._binnings)
        self._n += 1
        self._sumw += w
        self._sumw2 += w**2
        self._data.append(x)
        self._weights.append([w])

    def numpy(self):
        if len(self._binnings) == 1:
            return numpy.histogram(self._data, bins=self._binnings[0].num(self._data), range=self._binnings[0].range(self._data), weights=self._weights)
        else:
            return numpy.histogramdd(self._data, bins=[x.num(self._data) for x in self._binnings], range=[x.range(self._data) for x in self._binnings], weights=self._weights)

################################################################################ functions

fcns = SymbolTable()

fcns["pi"] = data.ValueInstance(math.pi, None, index.DerivedColKey(parser.Literal(math.pi)))
fcns["e"] = data.ValueInstance(math.e, None, index.DerivedColKey(parser.Literal(math.e)))

class NumericalFunction:
    def __init__(self, name, fcn):
        self._name, self._fcn = name, fcn

    def __call__(self, node, symbols, counter, weight, rowkey):
        args = [runstep(x, symbols, counter, weight, rowkey) for x in node.arguments]
        for x in args:
            assert isinstance(x, data.Instance)
            if not (isinstance(x, data.ValueInstance) and isinstance(x.value, (int, float))):
                raise parser.LanguageError("all arguments in {0} must be numbers (not lists or records)".format(self._name), x.line, x.source)

        try:
            result = self._fcn(*[x.value for x in args])
        except Exception as err:
            raise parser.LanguageError(str(err), node.line, node.source)

        return data.ValueInstance(result, rowkey, index.DerivedColKey(node))

fcns["+"] = NumericalFunction("addition", lambda x, y: x + y)
fcns["-"] = NumericalFunction("subtraction", lambda x, y: x - y)
fcns["*"] = NumericalFunction("multiplication", lambda x, y: x * y)
fcns["/"] = NumericalFunction("division", lambda x, y: x / float(y))
fcns["*1"] = NumericalFunction("identity", lambda x: x)
fcns["*-1"] = NumericalFunction("negation", lambda x: -x)
fcns["**"] = NumericalFunction("exponentiation", lambda x, y: x**y)
fcns[">"] = NumericalFunction("greater-than", lambda x, y: x > y)
fcns[">="] = NumericalFunction("greater-or-equal-to", lambda x, y: x >= y)
fcns["<"] = NumericalFunction("less-than", lambda x, y: x < y)
fcns["<="] = NumericalFunction("less-or-equal-to", lambda x, y: x <= y)
fcns["abs"] = NumericalFunction("abs", abs)
fcns["round"] = NumericalFunction("round", round)
fcns["ceil"] = NumericalFunction("ceil", math.ceil)
fcns["factorial"] = NumericalFunction("factorial", math.factorial)
fcns["floor"] = NumericalFunction("floor", math.floor)
fcns["gcd"] = NumericalFunction("gcd", math.gcd)
fcns["isinf"] = NumericalFunction("isinf", math.isinf)
fcns["isnan"] = NumericalFunction("isnan", math.isnan)
fcns["ldexp"] = NumericalFunction("ldexp", math.ldexp)
fcns["exp"] = NumericalFunction("exp", math.exp)
fcns["expm1"] = NumericalFunction("expm1", math.expm1)
fcns["log"] = NumericalFunction("log", math.log)
fcns["log1p"] = NumericalFunction("log1p", math.log1p)
fcns["log10"] = NumericalFunction("log10", math.log10)
fcns["pow"] = NumericalFunction("pow", math.pow)
fcns["sqrt"] = NumericalFunction("sqrt", math.sqrt)
fcns["arccos"] = NumericalFunction("arccos", math.acos)
fcns["arcsin"] = NumericalFunction("arcsin", math.asin)
fcns["arctan"] = NumericalFunction("arctan", math.atan)
fcns["arctan2"] = NumericalFunction("arctan2", math.atan2)
fcns["cos"] = NumericalFunction("cos", math.cos)
fcns["hypot"] = NumericalFunction("hypot", math.hypot)
fcns["sin"] = NumericalFunction("sin", math.sin)
fcns["tan"] = NumericalFunction("tan", math.tan)
fcns["degrees"] = NumericalFunction("degrees", math.degrees)
fcns["radians"] = NumericalFunction("radians", math.radians)
fcns["arccosh"] = NumericalFunction("arccosh", math.acosh)
fcns["arcsinh"] = NumericalFunction("arcsinh", math.asinh)
fcns["arctanh"] = NumericalFunction("arctanh", math.atanh)
fcns["cosh"] = NumericalFunction("cosh", math.cosh)
fcns["sinh"] = NumericalFunction("sinh", math.sinh)
fcns["tanh"] = NumericalFunction("tanh", math.tanh)
fcns["erf"] = NumericalFunction("erf", math.erf)
fcns["erfc"] = NumericalFunction("erfc", math.erfc)
fcns["gamma"] = NumericalFunction("gamma", math.gamma)
fcns["lgamma"] = NumericalFunction("lgamma", math.lgamma)

class EqualityFunction:
    def __init__(self, negated):
        self._negated = negated

    def _simplify(self, t):
        if isinstance(t, (int, float)):
            return float
        else:
            return t

    def _evaluate(self, node, left, right):
        if type(left) != type(right) or (type(left) is data.ValueInstance and self._simplify(type(left.value)) != self._simplify(type(right.value))):
            raise parser.LanguageError("left and right of an equality/inequality check must have the same types", node.line, node.source)

        if type(left) is data.ValueInstance:
            out = (left.value == right.value)

        elif type(left) is data.ListInstance:
            if len(left.value) == len(right.value):
                if self._negated:
                    for x, y in zip(left.value, right.value):
                        if self._evaluate(node, x, y):
                            return True
                    return False
                else:
                    for x, y in zip(left.value, right.value):
                        if not self._evaluate(node, x, y):
                            return False
                    return True
            else:
                out = False

        elif type(left) is data.RecordInstance:
            out = (left.row == right.row) and (left.col == right.col)

        if self._negated:
            return not out
        else:
            return out

    def __call__(self, node, symbols, counter, weight, rowkey):
        args = [runstep(x, symbols, counter, weight, rowkey) for x in node.arguments]
        assert len(args) == 2
        assert isinstance(args[0], data.Instance) and isinstance(args[1], data.Instance)
        return data.ValueInstance(self._evaluate(node, args[0], args[1]), rowkey, index.DerivedColKey(node))

fcns["=="] = EqualityFunction(False)
fcns["!="] = EqualityFunction(True)

class InclusionFunction(EqualityFunction):
    def _evaluate(self, node, left, right):
        if type(right) != data.ListInstance:
            raise parser.LanguageError("value to the right of 'in' must be a list", node.line, node.source)
        if self._negated:
            for x in right.value:
                if not EqualityFunction._evaluate(self, node, left, x):
                    return False
            return True
        else:
            for x in right.value:
                if EqualityFunction._evaluate(self, node, left, x):
                    return True
            return False

fcns["in"] = InclusionFunction(False)
fcns["not in"] = InclusionFunction(True)

class BooleanFunction:
    def __init__(self, name, fcn):
        self._name, self._fcn = name, fcn

    def __call__(self, node, symbols, counter, weight, rowkey):
        def iterate():
            for x in node.arguments:
                arg = runstep(x, symbols, counter, weight, rowkey)
                assert isinstance(arg, data.Instance)
                if not (isinstance(arg, data.ValueInstance) and isinstance(arg.value, bool)):
                    raise parser.LanguageError("arguments of '{0}' must be boolean".format(self._name), arg.line, arg.source)
                yield arg.value
        return data.ValueInstance(self._fcn(iterate()), rowkey, index.DerivedColKey(node))

fcns["and"] = BooleanFunction("and", all)
fcns["or"] = BooleanFunction("or", any)
fcns["not"] = BooleanFunction("not", lambda iterate: not next(iterate))

def ifthenelse(node, symbols, counter, weight, rowkey):
    predicate = runstep(node.arguments[0], symbols, counter, weight, rowkey)
    if not (isinstance(predicate, data.ValueInstance) and isinstance(predicate.value, bool)):
        raise parser.LanguageError("predicte of if/then/else must be boolean", node.arguments[0].line, node.source)
    if predicate.value:
        return runstep(node.arguments[1], symbols, counter, weight, rowkey)
    else:
        return runstep(node.arguments[2], symbols, counter, weight, rowkey)

fcns["if"] = ifthenelse

def where(node, symbols, counter, weight, rowkey):
    raise NotImplementedError

fcns["where"] = where

################################################################################ run

def runstep(node, symbols, counter, weight, rowkey):
    if isinstance(node, parser.Literal):
        return data.ValueInstance(node.value, None, index.DerivedColKey(node))

    elif isinstance(node, parser.Symbol):
        return symbols[node.symbol]

    elif isinstance(node, parser.Block):
        raise NotImplementedError(node)

    elif isinstance(node, parser.Call):
        function = runstep(node.function, symbols, counter, weight, rowkey)
        if callable(function):
            return function(node, symbols, counter, weight, rowkey)
        else:
            raise parser.LanguageError("not a function; cannot be called", node.line, node.source)

    elif isinstance(node, parser.GetItem):
        raise NotImplementedError(node)

    elif isinstance(node, parser.GetAttr):
        raise NotImplementedError(node)

    elif isinstance(node, parser.Assignment):
        symbols[node.symbol] = runstep(node.expression, symbols, counter, weight, rowkey)

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
            component = runstep(axis.expression, symbols, counter, weight, rowkey)
            if isinstance(component, data.ValueInstance) and isinstance(component.value, (int, float)):
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

    output = dataset.newempty()
    counter = DirectoryCounter()
    for entry in dataset:
        if not isinstance(entry, data.RecordInstance):
            raise parser.LanguageError("entries must be records (outermost array structure must be RecordArray)")

        original = SymbolTable(fcns)
        for n in entry.fields():
            original[n] = entry[n]
        modified = SymbolTable(original)

        counter.fill(1.0)
        for node in source:
            runstep(node, modified, counter, 1.0, entry.row)

        if not modified.empty():
            out = entry.newempty()
            for n in modified:
                out[n] = modified[n]
            output.append(out)

    return output, counter

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
        "met": data.PrimitiveArray([100, 200, 300, 400]),
        "stuff": data.ListArray([0, 0, 1, 3], [0, 1, 3, 6], data.PrimitiveArray([1, 2, 2, 3, 3, 3]))
    })
    events.setindex()
    return data.instantiate(events)

def test_hist():
    output, counter = run(r"""
hist met
""", test_dataset())
    assert output.tolist() == []
    assert (counter.entries, counter.value, counter.error) == (4, 4.0, 2.0)
    assert counter.keys() == ["0"]
    counts, edges = counter["0"].numpy()
    assert counts.tolist() == [1, 0, 0, 1, 0, 0, 1, 0, 0, 1]
    assert edges.tolist() == [100, 130, 160, 190, 220, 250, 280, 310, 340, 370, 400]

def test_assign():
    output, counter = run(r"""
x = met
""", test_dataset())
    assert output.tolist() == [{"x": 100}, {"x": 200}, {"x": 300}, {"x": 400}]

def test_scalar():
    output, counter = run(r"""
x = met + 1
""", test_dataset())
    assert output.tolist() == [{"x": 101}, {"x": 201}, {"x": 301}, {"x": 401}]

    output, counter = run(r"""
x = met + 1 + met
""", test_dataset())
    assert output.tolist() == [{"x": 201}, {"x": 401}, {"x": 601}, {"x": 801}]

    output, counter = run(r"""
x = (met == met)
""", test_dataset())
    assert output.tolist() == [{"x": True}, {"x": True}, {"x": True}, {"x": True}]

    output, counter = run(r"""
x = (muons == muons)
""", test_dataset())
    assert output.tolist() == [{"x": True}, {"x": True}, {"x": True}, {"x": True}]

    output, counter = run(r"""
x = (met != met)
""", test_dataset())
    assert output.tolist() == [{"x": False}, {"x": False}, {"x": False}, {"x": False}]

    output, counter = run(r"""
x = (muons != muons)
""", test_dataset())
    assert output.tolist() == [{"x": False}, {"x": False}, {"x": False}, {"x": False}]

    output, counter = run(r"""
x = (stuff == stuff)
""", test_dataset())
    assert output.tolist() == [{"x": True}, {"x": True}, {"x": True}, {"x": True}]

    output, counter = run(r"""
x = (stuff != stuff)
""", test_dataset())
    assert output.tolist() == [{"x": False}, {"x": False}, {"x": False}, {"x": False}]

    output, counter = run(r"""
x = 1 in stuff
""", test_dataset())
    assert output.tolist() == [{"x": False}, {"x": True}, {"x": False}, {"x": False}]

    output, counter = run(r"""
x = 2 in stuff
""", test_dataset())
    assert output.tolist() == [{"x": False}, {"x": False}, {"x": True}, {"x": False}]

    output, counter = run(r"""
x = 1 not in stuff
""", test_dataset())
    assert output.tolist() == [{"x": True}, {"x": False}, {"x": True}, {"x": True}]

    output, counter = run(r"""
x = 2 not in stuff
""", test_dataset())
    assert output.tolist() == [{"x": True}, {"x": True}, {"x": False}, {"x": True}]

    output, counter = run(r"""
x = (met == met and met == met)
""", test_dataset())
    assert output.tolist() == [{"x": True}, {"x": True}, {"x": True}, {"x": True}]

    output, counter = run(r"""
x = (met == met and met != met)
""", test_dataset())
    assert output.tolist() == [{"x": False}, {"x": False}, {"x": False}, {"x": False}]

    output, counter = run(r"""
x = (met == met or met == met)
""", test_dataset())
    assert output.tolist() == [{"x": True}, {"x": True}, {"x": True}, {"x": True}]

    output, counter = run(r"""
x = (met != met or met == met)
""", test_dataset())
    assert output.tolist() == [{"x": True}, {"x": True}, {"x": True}, {"x": True}]

    output, counter = run(r"""
x = (not met == met)
""", test_dataset())
    assert output.tolist() == [{"x": False}, {"x": False}, {"x": False}, {"x": False}]

    output, counter = run(r"""
x = (not met != met)
""", test_dataset())
    assert output.tolist() == [{"x": True}, {"x": True}, {"x": True}, {"x": True}]

    output, counter = run(r"""
x = (if 1 in stuff then 1 else -1)
""", test_dataset())
    assert output.tolist() == [{"x": -1}, {"x": 1}, {"x": -1}, {"x": -1}]

    output, counter = run(r"""
x = (if 2 in stuff then "a" else "b")
""", test_dataset())
    assert output.tolist() == [{"x": "b"}, {"x": "b"}, {"x": "a"}, {"x": "b"}]
