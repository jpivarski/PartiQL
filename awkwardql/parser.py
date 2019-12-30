# A grammar heavily inspired by SQL, adapted for our purposes.

import lark


class QueryError(Exception):
    def __init__(self, message, line=None, source=None, defining=None):
        self.message, self.line = message, line
        if line is not None:
            message = "line {0}: {1}".format(line, message)
            if defining is not None:
                message = message + " (while defining macro {0})".format(repr(defining))
            if source is not None:
                context = source.split("\n")[line - 2:line + 1]
                if line == 1:
                    context[0] = "{0:>4d} --> {1}".format(line, context[0])
                    context[1:] = ["{0:>4d}     {1}".format(line + 1 + i, x) for i, x in enumerate(context[1:])]
                else:
                    context[0] = "{0:>4d}     {1}".format(line - 1, context[0])
                    context[1] = "{0:>4d} --> {1}".format(line, context[1])
                    context[2:] = ["{0:>4d}     {1}".format(line + 1 + i, x) for i, x in enumerate(context[2:])]
                message = message + "\nline\n" + "\n".join(context)
        super(QueryError, self).__init__(message)


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

expression: tabular

tabular:    minmaxby
minmaxby:   groupby    | minmaxby "min" "by" scalar -> minby
                       | minmaxby "max" "by" scalar -> maxby
groupby:    uniondiff  | groupby "group" "by" scalar
                       | groupby "group" "by" scalar "as" namelist
                       | groupby "group" "by" scalar "as" namelist "with" "{" blockitems "}" -> groupwith
                       | groupby "group" "by" scalar "as" namelist "to" "{" blockitems "}" -> groupto
                       | groupby "group" "by" scalar "as" namelist "to" scalar -> groupto
uniondiff:  intercross | uniondiff "union" intercross -> union
                       | uniondiff "except" intercross -> except
intercross: wherewith  | intercross "join" wherewith -> join
                       | intercross "cross" wherewith -> cross
wherewith:  pack       | wherewith "with" "{" blockitems "}" -> with
                       | wherewith "to" "{" blockitems "}" -> to
                       | wherewith "to" scalar -> to
                       | wherewith "where" scalar -> where
pack:       scalar     | scalar "as" namelist

scalar:     branch
branch:     or         | "if" or "then" branch ["else" branch]
or:         and        | or "or" and
and:        not        | and "and" not
not:        has        | "not" not -> isnot
has:        comparison | "has" namelist -> ishas
comparison: arith | arith "==" arith -> eq | arith "!=" arith -> ne
                  | arith ">" arith -> gt  | arith ">=" arith -> ge
                  | arith "<" arith -> lt  | arith "<=" arith -> le
                  | arith "in" expression -> in | arith "not" "in" expression -> notin
arith:   term     | arith "+" term  -> add | arith "-" term -> sub
term:    factor   | term "*" factor -> mul | term "/" factor -> div | term "%" factor -> mod
factor:  pow      | "+" factor      -> pos | "-" factor -> neg
pow:     call ["**" factor]
call:    atom     | call trailer
atom: "(" expression ")"
    | "{" blockitems "}" -> block
    | "[" exprlist? "]" -> newlist
    | CNAME -> symbol
    | "?" CNAME -> maybesymbol
    | INT -> int
    | FLOAT -> float
    | ESCAPED_STRING -> string

exprlist: expression ("," expression)*
namelist: CNAME | "(" CNAME ("," CNAME)* ")"
arglist:  expression ("," expression)*
trailer:  "(" arglist? ")" -> args
       |  "." CNAME -> attr
       |  "?." CNAME -> maybeattr
//     |  "[" arglist "]" -> items    // getitem would provide access to list order; let's work with pure sets

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
    fields = ()

    def __init__(self, *args, line=None, source=None):
        self.line, self.source = line, source
        for n, x in zip(self.fields, args):
            setattr(self, n, x)
            if self.line is None:
                if isinstance(x, list):
                    if len(x) != 0:
                        self.line = getattr(x[0], "line", None)
                else:
                    self.line = getattr(x, "line", None)
        if self.line is not None:
            assert self.source is not None
        self.validate()

    def validate(self):
        pass

    def __repr__(self):
        return "{0}({1})".format(type(self).__name__, ", ".join(repr(getattr(self, n)) for n in self.fields))

    def __eq__(self, other):
        return type(self) is type(other) and all(getattr(self, n) == getattr(other, n) for n in self.fields)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        def do(x):
            if isinstance(x, list):
                return tuple(x)
            else:
                return x
        return hash((type(self),) + tuple(do(getattr(self, n)) for n in self.fields))

    def replace(self, replacements):
        def do(x):
            if isinstance(x, list):
                return [do(y) for y in x]
            elif isinstance(x, AST):
                return x.replace(replacements)
            else:
                return x
        return type(self)(*[do(getattr(self, n)) for n in self.fields], line=self.line, source=self.source)

    def setnames(self, names):
        for n in self.fields:
            x = getattr(self, n)
            if isinstance(x, list):
                for y in x:
                    if isinstance(y, AST):
                        y.setnames(names)
            elif isinstance(x, AST):
                x.setnames(names)


class Statement(AST):
    pass


class BlockItem(Statement):
    pass


class Expression(BlockItem):
    pass


class Named:
    def setnames(self, names):
        if self.named is None:
            self.name = str(len(names))
        elif isinstance(self.named, Literal) and isinstance(self.named.value, str):
            self.name = self.named.value
        else:
            raise QueryError("name must be a constant string", self.line, self.source, None)
        if self.name in names:
            raise QueryError("name {0} already exists".format(repr(self.name)), self.line, self.source, None)
        names.add(self.name)

        subnames = set()
        for n in self.fields:
            x = getattr(self, n)
            if isinstance(x, list):
                for y in x:
                    y.setnames(subnames)
            elif isinstance(x, AST):
                x.setnames(subnames)


class Literal(Expression):
    fields = ("value",)


class Symbol(Expression):
    fields = ("symbol", "maybe")

    def __init__(self, symbol, maybe=False, line=None, source=None):
        super(Symbol, self).__init__(symbol, maybe, line=line, source=source)

    def __repr__(self):
        return "Symbol({0}{1})".format(repr(self.symbol), ", maybe=True" if self.maybe else "")

    def replace(self, replacements):
        return replacements.get(self.symbol, self)


class Block(Expression):
    fields = ("body",)

    def validate(self):
        if len(self.body) == 0:
            raise QueryError("expression block may not be empty", self.line, self.source, None)
        if not isinstance(self.body[-1], Expression):
            raise QueryError("in a block expression (temporary assignments inside curly"
                             "brackets), the last item (separated by semicolons or line"
                             "endings) must be an expression (something with a return"
                             "value)", self.body[-1].line, self.source, None)


class Call(Expression):
    fields = ("function", "arguments")

# class GetItem(Expression):
#     fields = ("container", "where")


class GetAttr(Expression):
    fields = ("object", "field", "maybe")

    def __repr__(self):
        return "GetAttr({0}, {1}{2})".format(repr(self.object), repr(self.field), ", maybe=True" if self.maybe else "")


class Pack(Expression):
    fields = ("container", "names")


class With(Expression):
    fields = ("container", "body", "new")


class Has(Expression):
    fields = ("names",)


class Assignment(BlockItem):
    fields = ("symbol", "expression")


class Histogram(Named, BlockItem):
    fields = ("axes", "weight", "named", "titled")


class Axis(AST):
    fields = ("expression", "binning")


class Vary(Statement):
    fields = ("trials", "body")

    def setnames(self, names):
        for x in self.trials:
            x.setnames(names)
        subnames = set()
        for x in self.body:
            x.setnames(subnames)


class Trial(Named, AST):
    fields = ("assignments", "named")


class Cut(Named, Statement):
    fields = ("expression", "weight", "named", "titled", "body")


def parse(source):
    start = parse.parser.parse(source)

    op2fcn = {"add": "+", "sub": "-", "mul": "*", "div": "/", "mod": "%", "pow": "**",
              "pos": "*1", "neg": "*-1",
              "eq": "==", "ne": "!=", "gt": ">", "ge": ">=", "lt": "<", "le": "<=",
              "in": ".in", "notin": ".not in",
              "and": ".and", "or": ".or", "isnot": ".not"}

    class Macro(Statement):
        fields = ("parameters", "body")

    class MacroBlock(AST):
        fields = ("body",)

    def getattributes(nodes, source, macros, defining):
        weight, named, titled = None, None, None
        for x in nodes:
            assert x.data == "attribute"
            x = x.children[0]
            if x.data == "weight":
                if weight is None:
                    weight = toast(x.children[0], macros, defining)
                else:
                    raise QueryError("weight may not be defined more than once", x.line, source, defining)
            if x.data == "named":
                if named is None:
                    named = toast(x.children[0], macros, defining)
                else:
                    raise QueryError("name may not be defined more than once", x.line, source, defining)
            if x.data == "titled":
                if titled is None:
                    titled = toast(x.children[0], macros, defining)
                else:
                    raise QueryError("title may not be defined more than once", x.line, source, defining)
        return weight, named, titled

    def toast(node, macros, defining):
        if isinstance(node, lark.Token):
            return None

        elif node.data == "macro":
            name = str(node.children[0])
            macros[name] = ([str(x) for x in node.children[1:-1]], MacroBlock(toast(node.children[-1], macros, name), source=source))
            return None

        elif node.data in ("symbol", "maybesymbol"):
            if str(node.children[0]) in macros or str(node.children[0]) == defining:
                raise QueryError("the name {0} should not be used as a variable and a macro".format(repr(str(node.children[0]))),
                                 node.children[0].line, source, defining)
            return Symbol(str(node.children[0]), node.data == "maybesymbol", line=node.children[0].line, source=source)

        elif node.data == "int":
            return Literal(int(str(node.children[0])), line=node.children[0].line, source=source)

        elif node.data == "float":
            return Literal(float(str(node.children[0])), line=node.children[0].line, source=source)

        elif node.data == "string":
            return Literal(eval(str(node.children[0])), line=node.children[0].line, source=source)

        elif (node.data in ("add", "sub", "mul", "div", "mod", "pow", "eq", "ne", "gt",
                            "ge", "lt", "le", "in", "notin", "and", "or") and
              len(node.children) == 2):
            return Call(Symbol(op2fcn[node.data]), [toast(node.children[0], macros, defining), toast(node.children[1], macros, defining)], source=source)

        elif node.data in ("pos", "neg", "isnot") and len(node.children) == 1:
            return Call(Symbol(op2fcn[node.data]), [toast(node.children[0], macros, defining)], source=source)

        elif node.data == "ishas":
            return Has(toast(node.children[0], macros, defining), source=source)

        elif node.data == "branch" and len(node.children) == 2:
            return Call(Symbol(".if"), [toast(node.children[0], macros, defining), toast(node.children[1], macros, defining)], source=source)

        elif node.data == "branch" and len(node.children) == 3:
            return Call(Symbol(".if"),
                        [toast(node.children[0], macros, defining),
                         toast(node.children[1], macros, defining),
                         toast(node.children[2], macros, defining)],
                        source=source)

        elif node.data == "call" and len(node.children) == 2:
            if node.children[1].data == "attr":
                return GetAttr(toast(node.children[0], macros, defining), str(node.children[1].children[0]), False, source=source)
            elif node.children[1].data == "maybeattr":
                return GetAttr(toast(node.children[0], macros, defining), str(node.children[1].children[0]), True, source=source)

            args = [toast(x, macros, defining) for x in node.children[1].children[0].children] if len(node.children[1].children) != 0 else []

            if node.children[1].data == "args":
                if(len(node.children[0].children) == 1 and
                   node.children[0].children[0].data == "symbol" and
                   str(node.children[0].children[0].children[0]) in macros):
                    name = str(node.children[0].children[0].children[0])
                    params, body = macros[name]
                    if len(params) != len(args):
                        raise QueryError("macro {0} has {1} parameters but {2} arguments"
                                         "were passed".format(repr(name),
                                                              len(params),
                                                              len(args)),
                                         node.children[0].children[0].children[0].line,
                                         source, defining)
                    return body.replace(dict(zip(params, args)))

                else:
                    return Call(toast(node.children[0], macros, defining), args, source=source)

            # elif node.children[1].data == "items":
            #     return GetItem(toast(node.children[0], macros, defining), args, source=source)

            else:
                assert False

        elif node.data == "namelist":
            return [str(x) for x in node.children]

        elif node.data == "pack" and len(node.children) == 2:
            return Pack(toast(node.children[0], macros, defining), toast(node.children[1], macros, defining), source=source)

        elif node.data == "with" and len(node.children) == 2:
            return With(toast(node.children[0], macros, defining), toast(node.children[1], macros, defining), False, source=source)

        elif node.data == "to" and len(node.children) == 2:
            return With(toast(node.children[0], macros, defining), toast(node.children[1], macros, defining), True, source=source)

        elif node.data == "where" and len(node.children) == 2:
            return Call(Symbol(".where"), [toast(node.children[0], macros, defining), toast(node.children[1], macros, defining)], source=source)

        elif node.data in ("join", "cross", "union", "except") and len(node.children) == 2:
            return Call(Symbol("." + node.data), [toast(node.children[0], macros, defining), toast(node.children[1], macros, defining)], source=source)

        elif node.data == "groupby" and len(node.children) == 2:
            return Call(Symbol(".group"), [toast(node.children[0], macros, defining), toast(node.children[1], macros, defining)], source=source)

        elif node.data == "groupby" and len(node.children) == 3:
            return Pack(Call(Symbol(".group"),
                             [toast(node.children[0], macros, defining),
                              toast(node.children[1], macros, defining)],
                             source=source),
                        toast(node.children[2], macros, defining),
                        source=source)

        elif node.data == "groupwith" and len(node.children) == 4:
            return With(Pack(Call(Symbol(".group"),
                                  [toast(node.children[0], macros, defining),
                                   toast(node.children[1], macros, defining)],
                                  source=source),
                             toast(node.children[2], macros, defining),
                             source=source),
                        toast(node.children[3], macros, defining),
                        False,
                        source=source)

        elif node.data == "groupto" and len(node.children) == 4:
            return With(Pack(Call(Symbol(".group"),
                                  [toast(node.children[0], macros, defining),
                                   toast(node.children[1], macros, defining)],
                                  source=source),
                             toast(node.children[2], macros, defining),
                             source=source),
                        toast(node.children[3], macros, defining),
                        True,
                        source=source)

        elif node.data == "minby" and len(node.children) == 2:
            return Call(Symbol(".min"), [toast(node.children[0], macros, defining), toast(node.children[1], macros, defining)], source=source)

        elif node.data == "maxby" and len(node.children) == 2:
            return Call(Symbol(".max"), [toast(node.children[0], macros, defining), toast(node.children[1], macros, defining)], source=source)

        elif node.data == "assignment":
            return Assignment(str(node.children[0]), toast(node.children[1], macros, defining), line=node.children[0].line, source=source)

        elif node.data == "block":
            return Block(toast(node.children[0], macros, defining), source=source)

        elif node.data == "newlist":
            return Call(Symbol("newlist"), list() if len(node.children) == 0 else toast(node.children[0], macros, defining), source=source)

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

        elif len(node.children) == 1 and node.data in ("statement", "blockitem",
                                                       "expression", "tabular",
                                                       "minmaxby", "groupby",
                                                       "uniondiff", "intercross",
                                                       "wherewith", "pack", "scalar",
                                                       "branch", "or", "and", "not",
                                                       "has", "comparison", "arith",
                                                       "term", "factor", "pow", "call",
                                                       "atom"):
            out = toast(node.children[0], macros, defining)
            if isinstance(out, MacroBlock) and len(out.body) == 1:
                return out.body[0]
            elif isinstance(out, MacroBlock):
                return Block(out.body, line=out.line, source=source)
            else:
                return out

        elif node.data in ("start", "statements", "blockitems", "exprlist", "assignments"):
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
