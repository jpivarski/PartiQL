# A grammar heavily inspired by SQL, adapted for our purposes.

import lark

class LanguageError(Exception):
    def __init__(self, message, line=None, source=None, defining=None):
        self.message, self.line = message, line
        if line is not None:
            message = "line {0}: {1}".format(line, message)
            if defining is not None:
                message = message + " (while defining macro {0})".format(repr(defining))
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
start:       NEWLINE? (statement  (NEWLINE | ";"+))* statement? NEWLINE? ";"* NEWLINE?
statements:  NEWLINE? (statement  (NEWLINE | ";"+))* statement  NEWLINE? ";"* NEWLINE?
blockitems:  NEWLINE? (blockitem  (NEWLINE | ";"+))* blockitem  NEWLINE?
assignments: NEWLINE? (assignment (NEWLINE | ";"+))* assignment NEWLINE?
statement:   macro | expression | assignment | histogram | vary | cut
blockitem:   macro | expression | assignment | histogram

macro:      "def" CNAME "(" [CNAME ("," CNAME)*] ")" "{" statements "}"
assignment: CNAME "=" expression

cut:        "cut" expression attribute* "{" statements "}"
vary:       "vary" trial+ "{" statements "}"
histogram:  "hist" axes attribute*

trial:      "by" "{" assignments "}" named?
attribute:  weight | named | titled
titled:     "titled" expression
named:      "named" expression
weight:     "weight" "by" expression
axes:       axis ("," axis)*
axis:       expression ["by" expression]

expression: scalar | tabular

tabular:    minmaxby
minmaxby:   groupby | groupby "min" "by" scalar -> minby | groupby "max" "by" scalar -> maxby
groupby:    fields  | fields "group" "by" where
fields:     where   | where "{" blockitems "}"
where:      union   | union "where" scalar
union:      cross   | cross "union" cross
cross:      join    | join "cross" join
join:       choose  | choose "join" choose
choose:     scalar  | namelist "from" scalar

scalar:     branch
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

namelist: CNAME | "(" CNAME ("," CNAME)* ")"
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
        self._validate()

    def _validate(self):
        pass

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
        return type(self)(*[do(getattr(self, n)) for n in self._fields], line=self.line, source=self.source)

    def setnames(self, names):
        for n in self._fields:
            x = getattr(self, n)
            if isinstance(x, list):
                for y in x:
                    if isinstance(y, AST):
                        y.setnames(names)
            elif isinstance(x, AST):
                x.setnames(names)

class Statement(AST): pass
class BlockItem(Statement): pass
class Expression(BlockItem): pass

class Named:
    def setnames(self, names):
        if self.named is None:
            self.name = str(len(names))
        elif isinstance(self.named, Literal) and isinstance(self.named.value, str):
            self.name = self.named.value
        else:
            raise LanguageError("name must be a constant string", self.line, self.source, None)
        if self.name in names:
            raise LanguageError("name {0} already exists".format(repr(self.name)), x.line, self.source, None)
        names.add(self.name)

        names = set()
        for n in self._fields:
            x = getattr(self, n)
            if isinstance(x, list):
                for y in x:
                    y.setnames(names)
            elif isinstance(x, AST):
                x.setnames(names)

class Literal(Expression):
    _fields = ("value",)

class Symbol(Expression):
    _fields = ("symbol",)

    def replace(self, replacements):
        return replacements.get(self.symbol, self)

class Block(Expression):
    _fields = ("body",)

    def _validate(self):
        if len(self.body) == 0:
            raise LanguageError("expression block may not be empty", self.line, self.source, None)
        for x in self.body[:-1]:
            if isinstance(x, Expression) or not isinstance(x, BlockItem):
                raise LanguageError("every item in a block (surrounded by curly braces) must be an assignment or a histogram except the last one, which must be an expression with a return value", x.line, self.source, None)
        if not isinstance(self.body[-1], Expression):
            raise LanguageError("every item in a block (surrounded by curly braces) must be an assignment or a histogram except the last one, which must be an expression with a return value", self.body[-1].line, self.source, None)

class Call(Expression):
    _fields = ("function", "arguments")

class GetItem(Expression):
    _fields = ("container", "where")

class GetAttr(Expression):
    _fields = ("object", "field")

class Choose(Expression):
    _fields = ("symbols", "table")

class TableBlock(Expression):
    _fields = ("table", "body")

    def _validate(self):
        for x in self.body:
            if isinstance(x, Expression) or not isinstance(x, BlockItem):
                raise LanguageError("every item in a block defining a table (curly braces after a table definitino) must be an assignment or a histogram", x.line, self.source, None)

class GroupBy(Expression):
    _fields = ("table", "quantifier")

class MinMaxBy(Expression):
    _fields = ("table", "ismin", "quantifier")

class Assignment(BlockItem):
    _fields = ("symbol", "expression")

class Histogram(Named, BlockItem):
    _fields = ("axes", "weight", "named", "titled")

class Axis(AST):
    _fields = ("expression", "binning")

class Vary(Statement):
    _fields = ("trials", "body")

    def _validate(self):
        for x in self.body:
            if isinstance(x, Expression) or not isinstance(x, Statement):
                raise LanguageError("every statement in a vary must be an assignment, a histogram, a vary, or a cut, not an expression", x.line, self.source, None)

class Trial(Named, AST):
    _fields = ("assignments", "named")

class Cut(Named, Statement):
    _fields = ("expression", "weight", "named", "titled", "body")

    def _validate(self):
        for x in self.body:
            if isinstance(x, Expression) or not isinstance(x, Statement):
                raise LanguageError("every statement in a cut must be an assignment, a histogram, a vary, or a cut, not an expression", x.line, self.source, None)

def parse(source):
    start = parse.parser.parse(source)

    op2fcn = {"add": "+", "sub": "-", "mul": "*", "div": "/", "pow": "**",
              "pos": "*1", "neg": "*-1",
              "eq": "==", "ne": "!=", "gt": ">", "ge": ">=", "lt": "<", "le": "<=", "in": "in", "notin": "not in",
              "and": "and", "or": "or", "isnot": "not"}

    class Macro(Statement):
        _fields = ("parameters", "body")

    class MacroBlock(AST):
        _fields = ("body",)

    def getattributes(nodes, source, macros, defining):
        weight, named, titled = None, None, None
        for x in nodes:
            assert x.data == "attribute"
            x = x.children[0]
            if x.data == "weight":
                if weight is None:
                    weight = toast(x.children[0], macros, defining)
                else:
                    raise LanguageError("weight may not be defined more than once", x.line, source, defining)
            if x.data == "named":
                if named is None:
                    named = toast(x.children[0], macros, defining)
                else:
                    raise LanguageError("name may not be defined more than once", x.line, source, defining)
            if x.data == "titled":
                if titled is None:
                    titled = toast(x.children[0], macros, defining)
                else:
                    raise LanguageError("title may not be defined more than once", x.line, source, defining)
        return weight, named, titled

    def toast(node, macros, defining):
        if isinstance(node, lark.Token):
            return None

        elif node.data == "macro":
            name = str(node.children[0])
            macros[name] = ([str(x) for x in node.children[1:-1]], MacroBlock(toast(node.children[-1], macros, name), source=source))
            return None

        elif node.data == "symbol":
            if str(node.children[0]) in macros or str(node.children[0]) == defining:
                raise LanguageError("the name {0} should not be used as a variable and a macro".format(repr(str(node.children[0]))), node.children[0].line, source, defining)
            return Symbol(str(node.children[0]), line=node.children[0].line, source=source)
        elif node.data == "int":
            return Literal(int(str(node.children[0])), line=node.children[0].line, source=source)

        elif node.data == "float":
            return Literal(float(str(node.children[0])), line=node.children[0].line, source=source)

        elif node.data == "string":
            return Literal(eval(str(node.children[0])), line=node.children[0].line, source=source)

        elif node.data in ("add", "sub", "mul", "div", "pow", "eq", "ne", "gt", "ge", "lt", "le", "in", "notin", "and", "or") and len(node.children) == 2:
            return Call(Symbol(op2fcn[node.data]), [toast(node.children[0], macros, defining), toast(node.children[1], macros, defining)], source=source)

        elif node.data in ("pos", "neg", "isnot") and len(node.children) == 1:
            return Call(Symbol(op2fcn[node.data]), [toast(node.children[0], macros, defining)], source=source)

        elif node.data == "branch" and len(node.children) == 3:
            return Call(Symbol("if"), [toast(node.children[0], macros, defining), toast(node.children[1], macros, defining), toast(node.children[2], macros, defining)], source=source)

        elif node.data == "call" and len(node.children) == 2:
            if node.children[1].data == "attr":
                return GetAttr(toast(node.children[0], macros, defining), str(node.children[1].children[0]), source=source)

            args = [toast(x, macros, defining) for x in node.children[1].children[0].children] if len(node.children[1].children) != 0 else []

            if node.children[1].data == "args":
                if len(node.children[0].children) == 1 and node.children[0].children[0].data == "symbol" and str(node.children[0].children[0].children[0]) in macros:
                    name = str(node.children[0].children[0].children[0])
                    params, body = macros[name]
                    if len(params) != len(args):
                        raise LanguageError("macro {0} has {1} parameters but {2} arguments were passed".format(repr(name), len(params), len(args)), node.children[0].children[0].children[0].line, source, defining)
                    return body.replace(dict(zip(params, args)))

                else:
                    return Call(toast(node.children[0], macros, defining), args, source=source)

            elif node.children[1].data == "items":
                return GetItem(toast(node.children[0], macros, defining), args, source=source)

            else:
                assert False

        elif node.data == "choose" and len(node.children) == 2:
            return Choose(toast(node.children[0], macros, defining), toast(node.children[1], macros, defining), source=source)

        elif node.data == "namelist":
            return [str(x) for x in node.children]

        elif node.data in ("join", "cross", "union", "where") and len(node.children) == 2:
            return Call(Symbol(node.data), [toast(node.children[0], macros, defining), toast(node.children[1], macros, defining)], source=source)

        elif node.data == "fields" and len(node.children) == 2:
            return TableBlock(toast(node.children[0], macros, defining), toast(node.children[1], macros, defining), source=source)

        elif node.data == "groupby" and len(node.children) == 2:
            return GroupBy(toast(node.children[0], macros, defining), toast(node.children[1], macros, defining), source=source)

        elif node.data == "minby" and len(node.children) == 2:
            return MinMaxBy(toast(node.children[0], macros, defining), True, toast(node.children[1], macros, defining), source=source)

        elif node.data == "maxby" and len(node.children) == 2:
            return MinMaxBy(toast(node.children[0], macros, defining), False, toast(node.children[1], macros, defining), source=source)

        elif node.data == "assignment":
            return Assignment(str(node.children[0]), toast(node.children[1], macros, defining), line=node.children[0].line, source=source)

        elif node.data == "block":
            return Block(toast(node.children[0], macros, defining), source=source)

        elif node.data == "histogram":
            weight, named, titled = getattributes(node.children[1:], source, macros, defining)
            return Histogram(toast(node.children[0], macros, defining), weight, named, titled, source=source)

        elif node.data == "axes":
            return [toast(x, macros, defining) for x in node.children]

        elif node.data == "axis":
            if len(node.children) == 1:
                binning = None
            else:
                binning = toast(node.children[1], macros, defining)
            return Axis(toast(node.children[0], macros, defining), binning, source=source)

        elif node.data == "vary":
            return Vary([toast(x, macros, defining) for x in node.children[:-1]], toast(node.children[-1], macros, defining), source=source)

        elif node.data == "trial":
            if len(node.children) == 1:
                named = None
            else:
                named = toast(node.children[1].children[0], macros, defining)
            return Trial(toast(node.children[0], macros, defining), named, source=source)

        elif node.data == "cut":
            weight, named, titled = getattributes(node.children[1:-1], source, macros, defining)
            return Cut(toast(node.children[0], macros, defining), weight, named, titled, toast(node.children[-1], macros, defining), source=source)

        elif len(node.children) == 1 and node.data in ("statement", "blockitem", "expression", "tabular", "minmaxby", "groupby", "fields", "where", "union", "cross", "join", "choose", "scalar", "branch", "or", "and", "not", "comparison", "arith", "term", "factor", "pow", "call", "atom"):
            out = toast(node.children[0], macros, defining)
            if isinstance(out, MacroBlock) and len(out.body) == 1:
                return out.body[0]
            elif isinstance(out, MacroBlock):
                return Block(out.body, line=out.line, source=source)
            else:
                return out

        elif node.data in ("start", "statements", "blockitems", "assignments"):
            if node.data == "statements":
                macros = dict(macros)
            out = []
            for x in node.children:
                if not isinstance(x, lark.Token):
                    y = toast(x, macros, defining)
                    if y is None:
                        pass
                    elif isinstance(y, MacroBlock):
                        out.extend(y.body)
                    else:
                        out.append(y)
            return out

        else:
            raise NotImplementedError("node: {0} numchildren: {1}\n{2}".format(node.data, len(node.children), node.pretty()))

    out = toast(start, {}, None)
    if not parse.debugging:
        if len(out) == 0:
            raise LanguageError("source may not be empty", 1, source, None)
        for x in out:
            if isinstance(x, Expression) or not isinstance(x, Statement):
                raise LanguageError("every statement must be an assignment, a histogram, a vary, or a cut, not an expression", x.line, source, None)

    names = set()
    for x in out:
        x.setnames(names)

    return out

parse.parser = lark.Lark(grammar)
parse.debugging = False

################################################################################ tests

def test_whitespace():
    parse.debugging = True
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
    parse.debugging = False

def test_expressions():
    parse.debugging = True
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
    parse.debugging = False

def test_assign():
    parse.debugging = True
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
    parse.debugging = False

def test_table():
    parse.debugging = True
    assert parse(r"x from table") == [Choose(["x"], Symbol("table"))]
    assert parse(r"(x, y) from table") == [Choose(["x", "y"], Symbol("table"))]
    assert parse(r"f((x, y) from table)") == [Call(Symbol("f"), [Choose(["x", "y"], Symbol("table"))])]
    assert parse(r"f(x, y from table)") == [Call(Symbol("f"), [Symbol("x"), Choose(["y"], Symbol("table"))])]
    assert parse(r"x from X join y from Y") == [Call(Symbol("join"), [Choose(["x"], Symbol("X")), Choose(["y"], Symbol("Y"))])]
    assert parse(r"x from X cross y from Y") == [Call(Symbol("cross"), [Choose(["x"], Symbol("X")), Choose(["y"], Symbol("Y"))])]
    assert parse(r"x from X union y from Y") == [Call(Symbol("union"), [Choose(["x"], Symbol("X")), Choose(["y"], Symbol("Y"))])]
    assert parse(r"x from X where x > 0") == [Call(Symbol("where"), [Choose(["x"], Symbol("X")), Call(Symbol(">"), [Symbol("x"), Literal(0)])])]
    assert parse(r"x from X cross y from Y join z from Z") == [Call(Symbol("cross"), [Choose(["x"], Symbol("X")), Call(Symbol("join"), [Choose(["y"], Symbol("Y")), Choose(["z"], Symbol("Z"))])])]
    assert parse(r"(x from X cross y from Y) join z from Z") == [Call(Symbol("join"), [Call(Symbol("cross"), [Choose(["x"], Symbol("X")), Choose(["y"], Symbol("Y"))]), Choose(["z"], Symbol("Z"))])]
    assert parse(r"x from X union y from Y cross z from Z") == [Call(Symbol("union"), [Choose(["x"], Symbol("X")), Call(Symbol("cross"), [Choose(["y"], Symbol("Y")), Choose(["z"], Symbol("Z"))])])]
    assert parse(r"(x from X union y from Y) cross z from Z") == [Call(Symbol("cross"), [Call(Symbol("union"), [Choose(["x"], Symbol("X")), Choose(["y"], Symbol("Y"))]), Choose(["z"], Symbol("Z"))])]
    assert parse(r"(x from X where x > 0) union y from Y") == [Call(Symbol("union"), [Call(Symbol("where"), [Choose(["x"], Symbol("X")), Call(Symbol(">"), [Symbol("x"), Literal(0)])]), Choose(["y"], Symbol("Y"))])]
    assert parse(r"x from table {y = x}") == [TableBlock(Choose(["x"], Symbol("table")), [Assignment("y", Symbol("x"))])]
    assert parse(r"x from table where x > 0 {y = x}") == [TableBlock(Call(Symbol("where"), [Choose(["x"], Symbol("table")), Call(Symbol(">"), [Symbol("x"), Literal(0)])]), [Assignment("y", Symbol("x"))])]
    assert parse(r"x from table group by table") == [GroupBy(Choose(["x"], Symbol("table")), Symbol("table"))]
    assert parse(r"x from table where x > 0 group by table") == [GroupBy(Call(Symbol("where"), [Choose(["x"], Symbol("table")), Call(Symbol(">"), [Symbol("x"), Literal(0)])]), Symbol("table"))]
    assert parse(r"x from table {y = x} group by table") == [GroupBy(TableBlock(Choose(["x"], Symbol("table")), [Assignment("y", Symbol("x"))]), Symbol("table"))]
    assert parse(r"x from table where x > 0 {y = x} group by table") == [GroupBy(TableBlock(Call(Symbol("where"), [Choose(["x"], Symbol("table")), Call(Symbol(">"), [Symbol("x"), Literal(0)])]), [Assignment("y", Symbol("x"))]), Symbol("table"))]
    assert parse(r"x from table min by x") == [MinMaxBy(Choose(["x"], Symbol("table")), True, Symbol("x"))]
    assert parse(r"x from table max by x") == [MinMaxBy(Choose(["x"], Symbol("table")), False, Symbol("x"))]
    assert parse(r"x from table group by table min by x") == [MinMaxBy(GroupBy(Choose(["x"], Symbol("table")), Symbol("table")), True, Symbol("x"))]
    parse.debugging = False

def test_histogram():
    assert parse(r"hist pt") == [Histogram([Axis(Symbol("pt"), None)], None, None, None)]
    assert parse(r"hist pt, eta") == [Histogram([Axis(Symbol("pt"), None), Axis(Symbol("eta"), None)], None, None, None)]
    assert parse(r"hist pt by regular(100, 0, 150)") == [Histogram([Axis(Symbol("pt"), Call(Symbol("regular"), [Literal(100), Literal(0), Literal(150)]))], None, None, None)]
    assert parse(r"hist pt by regular(100, 0, 150), eta by regular(100, -5, 5)") == [Histogram([Axis(Symbol("pt"), Call(Symbol("regular"), [Literal(100), Literal(0), Literal(150)])), Axis(Symbol("eta"), Call(Symbol("regular"), [Literal(100), Call(Symbol("*-1"), [Literal(5)]), Literal(5)]))], None, None, None)]
    assert parse(r"hist pt weight by w") == [Histogram([Axis(Symbol("pt"), None)], Symbol("w"), None, None)]
    assert parse(r"hist pt, eta weight by w") == [Histogram([Axis(Symbol("pt"), None), Axis(Symbol("eta"), None)], Symbol("w"), None, None)]
    assert parse(r"hist pt by regular(100, 0, 150), eta weight by w") == [Histogram([Axis(Symbol("pt"), Call(Symbol("regular"), [Literal(100), Literal(0), Literal(150)])), Axis(Symbol("eta"), None)], Symbol("w"), None, None)]

    assert parse(r'hist pt named "hello"') == [Histogram([Axis(Symbol("pt"), None)], None, Literal("hello"), None)]
    assert parse(r'hist pt, eta named "hello"') == [Histogram([Axis(Symbol("pt"), None), Axis(Symbol("eta"), None)], None, Literal("hello"), None)]
    assert parse(r'hist pt weight by w named "hello"') == [Histogram([Axis(Symbol("pt"), None)], Symbol("w"), Literal("hello"), None)]
    assert parse(r'hist pt by regular(100, 0, 150) named "hello"') == [Histogram([Axis(Symbol("pt"), Call(Symbol("regular"), [Literal(100), Literal(0), Literal(150)]))], None, Literal("hello"), None)]
    assert parse(r'hist pt by regular(100, 0, 150) weight by w named "hello"') == [Histogram([Axis(Symbol("pt"), Call(Symbol("regular"), [Literal(100), Literal(0), Literal(150)]))], Symbol("w"), Literal("hello"), None)]

    assert parse(r'hist pt titled "there"') == [Histogram([Axis(Symbol("pt"), None)], None, None, Literal("there"))]
    assert parse(r'hist pt, eta titled "there"') == [Histogram([Axis(Symbol("pt"), None), Axis(Symbol("eta"), None)], None, None, Literal("there"))]
    assert parse(r'hist pt weight by w titled "there"') == [Histogram([Axis(Symbol("pt"), None)], Symbol("w"), None, Literal("there"))]
    assert parse(r'hist pt by regular(100, 0, 150) titled "there"') == [Histogram([Axis(Symbol("pt"), Call(Symbol("regular"), [Literal(100), Literal(0), Literal(150)]))], None, None, Literal("there"))]
    assert parse(r'hist pt by regular(100, 0, 150) weight by w titled "there"') == [Histogram([Axis(Symbol("pt"), Call(Symbol("regular"), [Literal(100), Literal(0), Literal(150)]))], Symbol("w"), None, Literal("there"))]

    assert parse(r'hist pt named "hello" titled "there"') == [Histogram([Axis(Symbol("pt"), None)], None, Literal("hello"), Literal("there"))]
    assert parse(r'hist pt, eta named "hello" titled "there"') == [Histogram([Axis(Symbol("pt"), None), Axis(Symbol("eta"), None)], None, Literal("hello"), Literal("there"))]
    assert parse(r'hist pt weight by w named "hello" titled "there"') == [Histogram([Axis(Symbol("pt"), None)], Symbol("w"), Literal("hello"), Literal("there"))]
    assert parse(r'hist pt by regular(100, 0, 150) named "hello" titled "there"') == [Histogram([Axis(Symbol("pt"), Call(Symbol("regular"), [Literal(100), Literal(0), Literal(150)]))], None, Literal("hello"), Literal("there"))]
    assert parse(r'hist pt by regular(100, 0, 150) weight by w named "hello" titled "there"') == [Histogram([Axis(Symbol("pt"), Call(Symbol("regular"), [Literal(100), Literal(0), Literal(150)]))], Symbol("w"), Literal("hello"), Literal("there"))]

def test_cutvary():
    assert parse(r"""
cut x > 0 {
    hist x
}
""") == [Cut(Call(Symbol(">"), [Symbol("x"), Literal(0)]), None, None, None, [Histogram([Axis(Symbol('x'), None)], None, None, None)])]
    assert parse(r"""
cut x > 0 {
    hist x
}
cut x <= 0 {
    hist x
}
""") == [Cut(Call(Symbol(">"), [Symbol("x"), Literal(0)]), None, None, None, [Histogram([Axis(Symbol('x'), None)], None, None, None)]), Cut(Call(Symbol("<="), [Symbol("x"), Literal(0)]), None, None, None, [Histogram([Axis(Symbol('x'), None)], None, None, None)])]
    assert parse(r"""
cut x > 0 weight by w {
    hist x
}
""") == [Cut(Call(Symbol(">"), [Symbol("x"), Literal(0)]), Symbol("w"), None, None, [Histogram([Axis(Symbol('x'), None)], None, None, None)])]
    assert parse(r"""
cut x > 0 named "hello" {
    hist x
}
""") == [Cut(Call(Symbol(">"), [Symbol("x"), Literal(0)]), None, Literal("hello"), None, [Histogram([Axis(Symbol('x'), None)], None, None, None)])]
    assert parse(r"""
cut x > 0 weight by w named "hello" {
    hist x
}
""") == [Cut(Call(Symbol(">"), [Symbol("x"), Literal(0)]), Symbol("w"), Literal("hello"), None, [Histogram([Axis(Symbol('x'), None)], None, None, None)])]
    assert parse(r"""
cut x > 0 titled "there" {
    hist x
}
""") == [Cut(Call(Symbol(">"), [Symbol("x"), Literal(0)]), None, None, Literal("there"), [Histogram([Axis(Symbol('x'), None)], None, None, None)])]
    assert parse(r"""
cut x > 0 weight by w titled "there" {
    hist x
}
""") == [Cut(Call(Symbol(">"), [Symbol("x"), Literal(0)]), Symbol("w"), None, Literal("there"), [Histogram([Axis(Symbol('x'), None)], None, None, None)])]
    assert parse(r"""
cut x > 0 named "hello" titled "there" {
    hist x
}
""") == [Cut(Call(Symbol(">"), [Symbol("x"), Literal(0)]), None, Literal("hello"), Literal("there"), [Histogram([Axis(Symbol('x'), None)], None, None, None)])]
    assert parse(r"""
cut x > 0 weight by w named "hello" titled "there" {
    hist x
}
""") == [Cut(Call(Symbol(">"), [Symbol("x"), Literal(0)]), Symbol("w"), Literal("hello"), Literal("there"), [Histogram([Axis(Symbol('x'), None)], None, None, None)])]
    assert parse(r"""
cut x > 0 {
    cut y > 0 {
        hist z
    }
}
""") == [Cut(Call(Symbol(">"), [Symbol("x"), Literal(0)]), None, None, None, [Cut(Call(Symbol(">"), [Symbol("y"), Literal(0)]), None, None, None, [Histogram([Axis(Symbol("z"), None)], None, None, None)])])]
    assert parse(r"""
vary by {epsilon = 0} {
    hist epsilon
}
""") == [Vary([Trial([Assignment("epsilon", Literal(0))], None)], [Histogram([Axis(Symbol("epsilon"), None)], None, None, None)])]
    assert parse(r"""
vary by {x = 0; y = 0} {
    hist x + y
}
""") == [Vary([Trial([Assignment("x", Literal(0)), Assignment("y", Literal(0))], None)], [Histogram([Axis(Call(Symbol("+"), [Symbol("x"), Symbol("y")]), None)], None, None, None)])]
    assert parse(r"""
vary by {epsilon = 0} named "hello" {
    hist epsilon
}
""") == [Vary([Trial([Assignment("epsilon", Literal(0))], Literal("hello"))], [Histogram([Axis(Symbol("epsilon"), None)], None, None, None)])]
    assert parse(r"""
vary by {epsilon = 0} by {epsilon = 0.001} {
    hist epsilon
}
""") == [Vary([Trial([Assignment("epsilon", Literal(0))], None), Trial([Assignment("epsilon", Literal(0.001))], None)], [Histogram([Axis(Symbol("epsilon"), None)], None, None, None)])]
    assert parse(r"""
vary by {epsilon = 0} named "one"
     by {epsilon = 0.001} {
    hist epsilon
}
""") == [Vary([Trial([Assignment("epsilon", Literal(0))], Literal("one")), Trial([Assignment("epsilon", Literal(0.001))], None)], [Histogram([Axis(Symbol("epsilon"), None)], None, None, None)])]
    assert parse(r"""
vary by {epsilon = 0} named "one"
     by {epsilon = 0.001} named "two" {
    hist epsilon
}
""") == [Vary([Trial([Assignment("epsilon", Literal(0))], Literal("one")), Trial([Assignment("epsilon", Literal(0.001))], Literal("two"))], [Histogram([Axis(Symbol("epsilon"), None)], None, None, None)])]
    assert parse(r"""
cut x > 0 {
    vary by {epsilon = 0} {
        hist epsilon
    }
}
""") == [Cut(Call(Symbol(">"), [Symbol("x"), Literal(0)]), None, None, None, [Vary([Trial([Assignment("epsilon", Literal(0))], None)], [Histogram([Axis(Symbol("epsilon"), None)], None, None, None)])])]
    assert parse(r"""
vary by {epsilon = 0} {
    cut x > 0 {
        hist epsilon
    }
}
""") == [Vary([Trial([Assignment("epsilon", Literal(0))], None)], [Cut(Call(Symbol(">"), [Symbol("x"), Literal(0)]), None, None, None, [Histogram([Axis(Symbol("epsilon"), None)], None, None, None)])])]

def test_macro():
    "Macros haven't been fully tested, but I'll leave that for later."
    assert parse(r"""
def f() {
    x
}
hist f()
""") == [Histogram([Axis(Symbol("x"), None)], None, None, None)]
    assert parse(r"""
def f() {
    hist x
}
f()
""") == [Histogram([Axis(Symbol("x"), None)], None, None, None)]
    assert parse(r"""
def f(y) {
    hist y
}
f(x)
""") == [Histogram([Axis(Symbol("x"), None)], None, None, None)]

def test_benchmark8():
    # https://github.com/iris-hep/adl-benchmarks-index/
    parse(r"""
leptons = electrons union muons

cut count(leptons) >= 3 {
    pair = (one, two) from electrons union (one, two) from muons
           where one.charge != two.charge
           min by abs(mass(one, two) - 91.2)

    third = x in leptons where x != pair.one and x != pair.two max by x.pt

    hist met      by regular(100, 0, 150) titled "transverse mass of the missing energy"
    hist third.pt by regular(100, 0, 150) titled "third lepton pt"
}
""")
