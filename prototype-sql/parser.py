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

expression: tabular | has
has:        "has" namelist

tabular:    minmaxby
minmaxby:   groupby    | minmaxby "min" "by" scalar -> minby | minmaxby "max" "by" scalar -> maxby
groupby:    union      | groupby "group" "by" scalar
union:      cross      | union "union" cross
cross:      join       | cross "cross" join
join:       where      | join "join" where
where:      with       | where "where" scalar
with:       pack       | with "with" "{" blockitems "}"
pack:       scalar     | scalar "as" namelist

scalar:     branch
branch:     or         | "if" or "then" branch ["else" branch]
or:         and        | or "or" and
and:        not        | and "and" not
not:        comparison | "not" not -> isnot
comparison: arith | arith "==" arith -> eq | arith "!=" arith -> ne
                  | arith ">" arith -> gt  | arith ">=" arith -> ge
                  | arith "<" arith -> lt  | arith "<=" arith -> le
                  | arith "in" expression -> in | arith "not" "in" expression -> notin
arith:   term     | arith "+" term  -> add | arith "-" term -> sub
term:    factor   | term "*" factor -> mul | term "/" factor -> div
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
        if not isinstance(self.body[-1], Expression):
            raise LanguageError("in a block expression (temporary assignments inside curly brackets), the last item (separated by semicolons or line endings) must be an expression (something with a return value)", self.body[-1].line, self.source, None)

class Call(Expression):
    _fields = ("function", "arguments")

class GetItem(Expression):
    _fields = ("container", "where")

class GetAttr(Expression):
    _fields = ("object", "field")

class Pack(Expression):
    _fields = ("container", "namelist")

class With(Expression):
    _fields = ("container", "body")

class Has(Expression):
    _fields = ("namelist",)

class Assignment(BlockItem):
    _fields = ("symbol", "expression")

class Histogram(Named, BlockItem):
    _fields = ("axes", "weight", "named", "titled")

class Axis(AST):
    _fields = ("expression", "binning")

class Vary(Statement):
    _fields = ("trials", "body")

class Trial(Named, AST):
    _fields = ("assignments", "named")

class Cut(Named, Statement):
    _fields = ("expression", "weight", "named", "titled", "body")

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

        elif node.data == "branch" and len(node.children) == 2:
            return Call(Symbol("if"), [toast(node.children[0], macros, defining), toast(node.children[1], macros, defining)], source=source)

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

        elif node.data == "namelist":
            return [str(x) for x in node.children]

        elif node.data == "pack" and len(node.children) == 2:
            return Pack(toast(node.children[0], macros, defining), toast(node.children[1], macros, defining), source=source)

        elif node.data == "with" and len(node.children) == 2:
            return With(toast(node.children[0], macros, defining), toast(node.children[1], macros, defining), source=source)

        elif node.data == "where" and len(node.children) == 2:
            return Call(Symbol("where"), [toast(node.children[0], macros, defining), toast(node.children[1], macros, defining)], source=source)

        elif node.data == "join" and len(node.children) == 2:
            return Call(Symbol("join"), [toast(node.children[0], macros, defining), toast(node.children[1], macros, defining)], source=source)

        elif node.data == "cross" and len(node.children) == 2:
            return Call(Symbol("cross"), [toast(node.children[0], macros, defining), toast(node.children[1], macros, defining)], source=source)

        elif node.data == "union" and len(node.children) == 2:
            return Call(Symbol("union"), [toast(node.children[0], macros, defining), toast(node.children[1], macros, defining)], source=source)

        elif node.data == "groupby" and len(node.children) == 2:
            return Call(Symbol("group"), [toast(node.children[0], macros, defining), toast(node.children[1], macros, defining)], source=source)

        elif node.data == "minby" and len(node.children) == 2:
            return Call(Symbol("min"), [toast(node.children[0], macros, defining), toast(node.children[1], macros, defining)], source=source)

        elif node.data == "maxby" and len(node.children) == 2:
            return Call(Symbol("max"), [toast(node.children[0], macros, defining), toast(node.children[1], macros, defining)], source=source)

        elif node.data == "has":
            return Has(toast(node.children[0], macros, defining), source=source)

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

        elif len(node.children) == 1 and node.data in ("statement", "blockitem", "expression", "tabular", "minmaxby", "groupby", "union", "cross", "join", "where", "with", "pack", "scalar", "branch", "or", "and", "not", "comparison", "arith", "term", "factor", "pow", "call", "atom"):
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

    names = set()
    for x in out:
        x.setnames(names)

    return out

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
    assert parse(r"x + y + z") == [Call(Symbol("+"), [Call(Symbol("+"), [Symbol("x"), Symbol("y")]), Symbol("z")])]
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
    assert parse(r"if p then if q then 1 else 2 else 3") == [Call(Symbol("if"), [Symbol("p"), Call(Symbol("if"), [Symbol("q"), Literal(1), Literal(2)]), Literal(3)])]
    assert parse(r"if p then { if q then 1 else 2 } else 3") == [Call(Symbol("if"), [Symbol("p"), Block([Call(Symbol("if"), [Symbol("q"), Literal(1), Literal(2)])]), Literal(3)])]
    assert parse(r"if p then 1 else if q then 2 else 3") == [Call(Symbol("if"), [Symbol("p"), Literal(1), Call(Symbol("if"), [Symbol("q"), Literal(2), Literal(3)])])]
    assert parse(r"if p then 1 else { if q then 2 else 3 }") == [Call(Symbol("if"), [Symbol("p"), Literal(1), Block([Call(Symbol("if"), [Symbol("q"), Literal(2), Literal(3)])])])]

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

def test_table():
    assert parse(r"table as x") == [Pack(Symbol("table"), ["x"])]
    assert parse(r"table as (x, y)") == [Pack(Symbol("table"), ["x", "y"])]
    assert parse(r"table with { x = 3 }") == [With(Symbol("table"), [Assignment("x", Literal(3))])]
    assert parse(r"table with { x = 3; y = x }") == [With(Symbol("table"), [Assignment("x", Literal(3)), Assignment("y", Symbol("x"))])]
    assert parse(r"table where x > 0") == [Call(Symbol("where"), [Symbol("table"), Call(Symbol(">"), [Symbol("x"), Literal(0)])])]
    assert parse(r"table with { x = 3 } where x > 0") == [Call(Symbol("where"), [With(Symbol("table"), [Assignment("x", Literal(3))]), Call(Symbol(">"), [Symbol("x"), Literal(0)])])]
    assert parse(r"a join b") == [Call(Symbol("join"), [Symbol("a"), Symbol("b")])]
    assert parse(r"a cross b") == [Call(Symbol("cross"), [Symbol("a"), Symbol("b")])]
    assert parse(r"a union b") == [Call(Symbol("union"), [Symbol("a"), Symbol("b")])]
    assert parse(r"a cross b join c") == [Call(Symbol("cross"), [Symbol("a"), Call(Symbol("join"), [Symbol("b"), Symbol("c")])])]
    assert parse(r"(a cross b) join c") == [Call(Symbol("join"), [Call(Symbol("cross"), [Symbol("a"), Symbol("b")]), Symbol("c")])]
    assert parse(r"a union b cross c") == [Call(Symbol("union"), [Symbol("a"), Call(Symbol("cross"), [Symbol("b"), Symbol("c")])])]
    assert parse(r"(a union b) cross c") == [Call(Symbol("cross"), [Call(Symbol("union"), [Symbol("a"), Symbol("b")]), Symbol("c")])]
    assert parse(r"a union b join c") == [Call(Symbol("union"), [Symbol("a"), Call(Symbol("join"), [Symbol("b"), Symbol("c")])])]
    assert parse(r"(a union b) join c") == [Call(Symbol("join"), [Call(Symbol("union"), [Symbol("a"), Symbol("b")]), Symbol("c")])]
    assert parse(r"a join b join c") == [Call(Symbol("join"), [Call(Symbol("join"), [Symbol("a"), Symbol("b")]), Symbol("c")])]
    assert parse(r"a cross b cross c") == [Call(Symbol("cross"), [Call(Symbol("cross"), [Symbol("a"), Symbol("b")]), Symbol("c")])]
    assert parse(r"a union b union c") == [Call(Symbol("union"), [Call(Symbol("union"), [Symbol("a"), Symbol("b")]), Symbol("c")])]
    assert parse(r"table group by x") == [Call(Symbol("group"), [Symbol("table"), Symbol("x")])]
    assert parse(r"table min by x") == [Call(Symbol("min"), [Symbol("table"), Symbol("x")])]
    assert parse(r"table max by x") == [Call(Symbol("max"), [Symbol("table"), Symbol("x")])]

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
