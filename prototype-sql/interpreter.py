# Simple, slow interpreter of row-wise data. Everything is a data.Instance.

import data
import parser

class SymbolTable:
    def __init__(self, parent=None, line=None, source=None):
        self._parent, self._line, self._source = parent, line, source
        self._table = {}

    def __repr__(self):
        return "<SymbolTable ({0} symbols) {1}>".format(len(self._table), repr(self._parent))

    def __getitem__(self, where):
        if where in self._table:
            return self._table[where]
        elif self._parent is not None:
            return self._parent[where]
        else:
            raise parser.LanguageError("unrecognized symbol: {0}".format(repr(where)), self._line, self._source)

    def __setitem__(self, where, what):
        self._table[where] = what

class Counter:
    def __init__(self):
        self._table = {}

    def __repr__(self):
        return "<Counter ({0} objects)>".format(len(self._table))

    def __getitem__(self, where):
        try:
            i = where.index("/")
        except ValueError:
            return self._table[where]
        else:
            return self._table[where[:i]][where[i + 1:]]

    def __setitem__(self, where, what):
        self._table[where] = what
