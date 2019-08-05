# A grammar heavily inspired by SQL, adapted for our purposes.

import lark

class LanguageError(Exception):
    def __init__(self, message, line):
        self.message, self.line = message, line
        if line is not None:
            message = "Line {0}: {1}".format(line, message)
        super(LanguageError, self).__init__(message)

grammar = r"""
start:       NEWLINE? (statement  (NEWLINE | ";"+))*            NEWLINE?
statements:  NEWLINE? (statement  (NEWLINE | ";"+))* statement  NEWLINE?
blockitems:  NEWLINE? (blockitem  (NEWLINE | ";"+))* blockitem  NEWLINE?
assignments: NEWLINE? (assignment (NEWLINE | ";"+))* assignment NEWLINE?
statement:   expression | assignment | histogram | vary | cut | macro
blockitem:   expression | assignment | histogram

macro:       "def" CNAME "(" [CNAME ("," CNAME)*] ")" "{" statements "}"
assignment:  CNAME "=" expression

cut:        "cut" expression weight? named? "{" statements "}"
vary:       "vary" trial+ "{" statements "}"
histogram:  "hist" expression ("by" arglist)* weight? named?

trial:  "by" "{" assignments "}"
named:  "named" expression
weight: "weight" "by" expression

expression: branch | groupby

groupby:    fields | fields "group" "by" where
fields:     where  | where "{" blockitems "}"
where:      union  | union "where" branch
union:      cross  | cross "union" cross
cross:      join   | join "cross" join
join:       choose | choose "join" choose
choose:     namelist "from" branch

branch:     or         | "if" or "then" or "else" or
or:         and        | and "or" and
and:        not        | not "and" not
not:        comparison | "not" not -> isnot
comparison: arith | arith "==" arith -> eq | arith "!=" arith -> ne
                  | arith ">" arith -> gt  | arith ">=" arith -> ge
                  | arith "<" arith -> lt  | arith "<=" arith -> le
                  | arith "in" expression -> in | arith "not" "in" expression -> in
arith:   term     | term "+" arith  -> add | term "-" arith -> sub
term:    factor   | factor "*" term -> mul | factor "/" term -> div
factor:  pow      | "+" factor      -> pos | "-" factor -> neg
pow:     call ["**" factor]
call:    atom     | call trailer
atom: "(" expression ")"
    | "{" blockitems "}" -> block
    | CNAME -> symbol
    | INT -> int
    | FLOAT -> float
    | ESCAPED_STRING -> string

namelist: CNAME ("," CNAME)*
arglist: expression ("," expression)*
trailer: "(" arglist? ")" -> args
       | "[" arglist "]" -> items
       | "." CNAME -> attr

COMMENT: "#" /.*/ NEWLINE | "//" /.*/ NEWLINE | "/*" /(.|\n|\r)*/ "*/"

%import common.CNAME
%import common.INT
%import common.FLOAT
%import common.ESCAPED_STRING
%import common.WS
%import common.NEWLINE

%ignore WS
%ignore COMMENT
"""

class AST:
    _fields = ()
    def __init__(self, *args, line=None):
        self.line = line
        for n, x in zip(self._fields, args):
            setattr(self, n, x)
            if self.line is None:
                if isinstance(x, list):
                    if len(x) != 0:
                        self.line = getattr(x[0], "line", None)
                else:
                    self.line = getattr(x, "line", None)

    def __repr__(self):
        return "{0}({1})".format(type(self).__name__, ", ".join(repr(getattr(self, n)) for n in self._fields))

    def __eq__(self, other):
        return type(self) is type(other) and all(getattr(self, n) == getattr(other, n) for n in self._fields)

    def __ne__(self, other):
        return not self.__eq__(other)

    def replace(self, replacements):
        def do(x):
            if isinstance(x, list):
                return [do(y) for y in x]
            elif isinstance(x, AST):
                return x.replace(replacements)
            else:
                return x
        return type(self)(*[do(getattr(self, n)) for n in self._fields], line=self.line)

class Literal(AST):
    _fields = ("value",)

class Symbol(AST):
    _fields = ("symbol",)

    def replace(self, replacements):
        return replacements.get(self.symbol, self)

class Block(AST):
    _fields = ("body",)

class Call(AST):
    _fields = ("function", "arguments")

class Choose(AST):
    _fields = ("symbols", "table")

class Assignment(AST):
    _fields = ("symbol", "expression")

class Histogram(AST):
    _fields = ("expression", "binning", "weight", "named")

class Vary(AST):
    _fields = ("trials", "statements")

class Cut(AST):
    _fields = ("expression", "weight", "named", "statements")

class Macro(AST):
    _fields = ("parameters", "body")

def parse(source, debug=False):
    start = parse.parser.parse(source)
    if debug:
        print(start.pretty())

    def toast(node, macros):
        if isinstance(node, lark.Token):
            return None

        elif node.data == "macro":
            macros[str(node.children[0])] = ([str(x) for x in node.children[1:-1]], Block(toast(node.children[-1], macros)))
            return None

        elif node.data == "symbol":
            if str(node.children[0]) in macros:
                raise LanguageError("the name {0} should not be used as a variable and a macro".format(repr(str(node.children[0]))), node.children[0].line)
            return Symbol(str(node.children[0]), line=node.children[0].line)

        elif node.data == "int":
            return Literal(int(str(node.children[0])), line=node.children[0].line)

        elif node.data == "float":
            return Literal(float(str(node.children[0])), line=node.children[0].line)

        elif node.data == "string":
            return Literal(eval(str(node.children[0])), line=node.children[0].line)

        elif node.data == "call" and len(node.children) == 2 and node.children[1].data == "args":
            args = [toast(x, macros) for x in node.children[1].children[0].children] if len(node.children[1].children) != 0 else []

            if len(node.children[0].children) == 1 and node.children[0].children[0].data == "symbol" and str(node.children[0].children[0].children[0]) in macros:
                name = str(node.children[0].children[0].children[0])
                params, body = macros[name]
                if len(params) != len(args):
                    raise LanguageError("macro {0} has {1} parameters but {2} arguments were passed".format(repr(name), len(params), len(args)), node.children[0].children[0].children[0].line)
                return body.replace(dict(zip(params, args)))

            else:
                return Call(toast(node.children[0], macros), args)

        elif node.data == "histogram":
            return Histogram(toast(node.children[0], macros), None, None, None)

        elif len(node.children) == 1 and node.data in ("statement", "blockitem", "expression", "branch", "or", "and", "not", "comparison", "arith", "term", "factor", "pow", "call"):
            return toast(node.children[0], macros)

        elif node.data in ("start", "statements", "blockitems", "assignments"):
            if node.data == "statements":
                macros = dict(macros)
            out = []
            for x in node.children:
                if not isinstance(x, lark.Token):
                    y = toast(x, macros)
                    if y is None:
                        pass
                    elif isinstance(y, Block):
                        out.extend(y.body)
                    else:
                        out.append(y)
            return out

        else:
            raise NotImplementedError("node: {0} numchildren: {1}\n{2}".format(node.data, len(node.children), node.pretty()))

    return toast(start, {})

parse.parser = lark.Lark(grammar)

################################################################################ tests

def test_parser():
    assert parse(r"""
def whatever(x) {
    hist x
}
whatever(pt)
""") == [Histogram(Symbol("pt"), None, None, None)]
