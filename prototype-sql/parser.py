# A grammar heavily inspired by SQL, adapted for our purposes.

import lark

class LanguageError(Exception):
    def __init__(self, message, line=None, source=None):
        self.message, self.line = message, line
        if line is not None:
            message = "line {0}: {1}".format(line, message)
            if source is not None:
                context = source.split("\n")[line - 2 : line + 1]
                if line == 1:
                    context[0]  = "{0:>4d} --> {1}".format(line, context[0])
                    context[1:] = ["{0:>4d}     {1}".format(line + 1 + i, x) for i, x in enumerate(context[1:])]
                else:
                    context[0]  = "{0:>4d}     {1}".format(line - 1, context[0])
                    context[1]  = "{0:>4d} --> {1}".format(line, context[1])
                    context[2:] = ["{0:>4d}     {1}".format(line + 1 + i, x) for i, x in enumerate(context[2:])]
                message = message + "\nline\n" + "\n".join(context)
        super(LanguageError, self).__init__(message)

grammar = r"""
start:       NEWLINE? (statement  (NEWLINE | ";"+))* statement? NEWLINE? ";"*  NEWLINE?
statements:  NEWLINE? (statement  (NEWLINE | ";"+))* statement NEWLINE? ";"*  NEWLINE?
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
                  | arith "in" expression -> in | arith "not" "in" expression -> notin
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

COMMENT: "#" /.*/ | "//" /.*/ | "/*" /(.|\n|\r)*/ "*/"

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

    def __init__(self, *args, line=None, source=None):
        self.line, self.source = line, source
        for n, x in zip(self._fields, args):
            setattr(self, n, x)
            if self.line is None:
                if isinstance(x, list):
                    if len(x) != 0:
                        self.line = getattr(x[0], "line", None)
                else:
                    self.line = getattr(x, "line", None)
        if self.line is not None:
            assert self.source is not None

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

class MacroBlock(AST):
    _fields = ("body",)

class Call(AST):
    _fields = ("function", "arguments")

class GetItem(AST):
    _fields = ("container", "where")

class GetAttr(AST):
    _fields = ("object", "field")

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

    op2fcn = {"add": "+", "sub": "-", "mul": "*", "div": "/", "pow": "**",
              "pos": "*1", "neg": "*-1",
              "eq": "==", "ne": "!=", "gt": ">", "ge": ">=", "lt": "<", "le": "<=", "in": "in", "notin": "not in",
              "and": "and", "or": "or", "isnot": "not"}

    def toast(node, macros):
        if isinstance(node, lark.Token):
            return None

        elif node.data == "macro":
            macros[str(node.children[0])] = ([str(x) for x in node.children[1:-1]], MacroBlock(toast(node.children[-1], macros), source=source))
            return None

        elif node.data == "symbol":
            if str(node.children[0]) in macros:
                raise LanguageError("the name {0} should not be used as a variable and a macro".format(repr(str(node.children[0]))), node.children[0].line, source)
            return Symbol(str(node.children[0]), line=node.children[0].line, source=source)
        elif node.data == "int":
            return Literal(int(str(node.children[0])), line=node.children[0].line, source=source)

        elif node.data == "float":
            return Literal(float(str(node.children[0])), line=node.children[0].line, source=source)

        elif node.data == "string":
            return Literal(eval(str(node.children[0])), line=node.children[0].line, source=source)

        elif node.data in ("add", "sub", "mul", "div", "pow", "eq", "ne", "gt", "ge", "lt", "le", "in", "notin", "and", "or") and len(node.children) == 2:
            return Call(Symbol(op2fcn[node.data]), [toast(node.children[0], macros), toast(node.children[1], macros)], source=source)

        elif node.data in ("pos", "neg", "isnot") and len(node.children) == 1:
            return Call(Symbol(op2fcn[node.data]), [toast(node.children[0], macros)], source=source)

        elif node.data == "branch" and len(node.children) == 3:
            return Call(Symbol("if"), [toast(node.children[0], macros), toast(node.children[1], macros), toast(node.children[2], macros)], source=source)

        elif node.data == "call" and len(node.children) == 2:
            if node.children[1].data == "attr":
                return GetAttr(toast(node.children[0], macros), str(node.children[1].children[0]), source=source)

            args = [toast(x, macros) for x in node.children[1].children[0].children] if len(node.children[1].children) != 0 else []

            if node.children[1].data == "args":
                if len(node.children[0].children) == 1 and node.children[0].children[0].data == "symbol" and str(node.children[0].children[0].children[0]) in macros:
                    name = str(node.children[0].children[0].children[0])
                    params, body = macros[name]
                    if len(params) != len(args):
                        raise LanguageError("macro {0} has {1} parameters but {2} arguments were passed".format(repr(name), len(params), len(args)), node.children[0].children[0].children[0].line, source)
                    return body.replace(dict(zip(params, args)))

                else:
                    return Call(toast(node.children[0], macros), args, source=source)

            elif node.children[1].data == "items":
                return GetItem(toast(node.children[0], macros), args, source=source)

            else:
                assert False

        elif node.data == "assignment":
            return Assignment(str(node.children[0]), toast(node.children[1], macros), line=node.children[0].line, source=source)

        elif node.data == "block":
            return Block(toast(node.children[0], macros), source=source)

        elif node.data == "histogram":
            return Histogram(toast(node.children[0], macros), None, None, None, source=source)

        elif len(node.children) == 1 and node.data in ("statement", "blockitem", "expression", "branch", "or", "and", "not", "comparison", "arith", "term", "factor", "pow", "call", "atom"):
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
                    elif isinstance(y, MacroBlock):
                        out.extend(y.body)
                    else:
                        out.append(y)
            return out

        else:
            raise NotImplementedError("node: {0} numchildren: {1}\n{2}".format(node.data, len(node.children), node.pretty()))

    return toast(start, {})

parse.parser = lark.Lark(grammar)

################################################################################ tests

def test_whitespace():
    assert parse(r"") == []
    assert parse(r"""x
""") == [Symbol("x")]
    assert parse(r"""
x""") == [Symbol("x")]
    assert parse(r"""
x
""") == [Symbol("x")]
    assert parse(r"""
x

""") == [Symbol("x")]
    assert parse(r"""

x
""") == [Symbol("x")]
    assert parse(r"""

x

""") == [Symbol("x")]
    assert parse(r"x   # comment") == [Symbol("x")]
    assert parse(r"x   // comment") == [Symbol("x")]
    assert parse(r"""
x  /* multiline
      comment */
""") == [Symbol("x")]
    assert parse(r"""# comment
x""") == [Symbol("x")]
    assert parse(r"""// comment
x""") == [Symbol("x")]
    assert parse(r"""/* multiline
                        comment */
x""") == [Symbol("x")]

def test_expressions():
    assert parse(r"x") == [Symbol("x")]
    assert parse(r"1") == [Literal(1)]
    assert parse(r"3.14") == [Literal(3.14)]
    assert parse(r'"hello"') == [Literal("hello")]
    assert parse(r"f(x)") == [Call(Symbol("f"), [Symbol("x")])]
    assert parse(r"f(x, 1, 3.14)") == [Call(Symbol("f"), [Symbol("x"), Literal(1), Literal(3.14)])]
    parse(r"a[0]")
    assert parse(r"a[0]") == [GetItem(Symbol("a"), [Literal(0)])]
    assert parse(r"a[0][i]") == [GetItem(GetItem(Symbol("a"), [Literal(0)]), [Symbol("i")])]
    assert parse(r"a[0, i]") == [GetItem(Symbol("a"), [Literal(0), Symbol("i")])]
    assert parse(r"a.b") == [GetAttr(Symbol("a"), "b")]
    assert parse(r"a.b.c") == [GetAttr(GetAttr(Symbol("a"), "b"), "c")]
    assert parse(r"x**2") == [Call(Symbol("**"), [Symbol("x"), Literal(2)])]
    assert parse(r"2*x") == [Call(Symbol("*"), [Literal(2), Symbol("x")])]
    assert parse(r"x/10") == [Call(Symbol("/"), [Symbol("x"), Literal(10)])]
    assert parse(r"x + y") == [Call(Symbol("+"), [Symbol("x"), Symbol("y")])]
    assert parse(r"x - y") == [Call(Symbol("-"), [Symbol("x"), Symbol("y")])]
    assert parse(r"x + 2*y") == [Call(Symbol("+"), [Symbol("x"), Call(Symbol("*"), [Literal(2), Symbol("y")])])]
    assert parse(r"(x + 2)*y") == [Call(Symbol("*"), [Call(Symbol("+"), [Symbol("x"), Literal(2)]), Symbol("y")])]
    assert parse(r"+x") == [Call(Symbol("*1"), [Symbol("x")])]
    assert parse(r"-x") == [Call(Symbol("*-1"), [Symbol("x")])]
    assert parse(r"+3.14") == [Call(Symbol("*1"), [Literal(3.14)])]
    assert parse(r"-3.14") == [Call(Symbol("*-1"), [Literal(3.14)])]
    assert parse(r"x == 0") == [Call(Symbol("=="), [Symbol("x"), Literal(0)])]
    assert parse(r"x != 0") == [Call(Symbol("!="), [Symbol("x"), Literal(0)])]
    assert parse(r"x > 0") == [Call(Symbol(">"), [Symbol("x"), Literal(0)])]
    assert parse(r"x >= 0") == [Call(Symbol(">="), [Symbol("x"), Literal(0)])]
    assert parse(r"x < 0") == [Call(Symbol("<"), [Symbol("x"), Literal(0)])]
    assert parse(r"x <= 0") == [Call(Symbol("<="), [Symbol("x"), Literal(0)])]
    assert parse(r"x in table") == [Call(Symbol("in"), [Symbol("x"), Symbol("table")])]
    assert parse(r"x not in table") == [Call(Symbol("not in"), [Symbol("x"), Symbol("table")])]
    assert parse(r"p and q") == [Call(Symbol("and"), [Symbol("p"), Symbol("q")])]
    assert parse(r"p or q") == [Call(Symbol("or"), [Symbol("p"), Symbol("q")])]
    assert parse(r"not p") == [Call(Symbol("not"), [Symbol("p")])]
    assert parse(r"p or q and r") == [Call(Symbol("or"), [Symbol("p"), Call(Symbol("and"), [Symbol("q"), Symbol("r")])])]
    assert parse(r"(p or q) and r") == [Call(Symbol("and"), [Call(Symbol("or"), [Symbol("p"), Symbol("q")]), Symbol("r")])]
    assert parse(r"if x > 0 then 1 else -1") == [Call(Symbol("if"), [Call(Symbol(">"), [Symbol("x"), Literal(0)]), Literal(1), Call(Symbol("*-1"), [Literal(1)])])]

def test_assign():
    assert parse(r"""
x = 5
x + 2
""") == [Assignment("x", Literal(5)), Call(Symbol("+"), [Symbol("x"), Literal(2)])]
    assert parse(r"""{
x = 5
x + 2
}""") == [Block([Assignment("x", Literal(5)), Call(Symbol("+"), [Symbol("x"), Literal(2)])])]
    assert parse(r"""
y = {
    x = 5
    x + 2
    }
y""") == [Assignment("y", Block([Assignment("x", Literal(5)), Call(Symbol("+"), [Symbol("x"), Literal(2)])])), Symbol("y")]
    assert parse(r"{x + 2}") == [Block([Call(Symbol("+"), [Symbol("x"), Literal(2)])])]
    assert parse(r"if x > 0 then {1} else {-1}") == [Call(Symbol("if"), [Call(Symbol(">"), [Symbol("x"), Literal(0)]), Block([Literal(1)]), Block([Call(Symbol("*-1"), [Literal(1)])])])]
