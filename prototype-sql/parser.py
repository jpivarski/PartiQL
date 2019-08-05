# A grammar heavily inspired by SQL, adapted for our purposes.

import lark

grammar = r"""
start: statements

statements:  NEWLINE? (statement  (NEWLINE | ";"+))* statement  NEWLINE?
sideeffects: NEWLINE? (sideeffect (NEWLINE | ";")+)* sideeffect NEWLINE?
assignments: NEWLINE? (assignment (NEWLINE | ";"+))* assignment NEWLINE?
statement:   assignment | histogram | vary | cut
sideeffect:  assignment | histogram

function:    "def" CNAME "(" namelist ")" "{" statements expression? "}"
assignment:  CNAME "=" (expression | table)

cut:        "cut" expression weight? named? "{" statements "}"
vary:       "vary" trial+ "{" statements "}"
histogram:  "hist" expression ("by" arglist)* weight? named?

trial:  "by" "{" assignments "}"
named:  "named" expression
weight: "weight" "by" expression

table:    groupby
groupby:  fields | fields "group" "by" where
fields:   where  | where "{" sideeffects "}"
where:    union  | union "where" expression
union:    cross  | cross "union" cross
cross:    join   | join "cross" join
join:     choose | choose "join" choose
choose:   namelist "from" tableatom
tableatom: "(" table ")" | CNAME -> symbol

namelist: CNAME ("," CNAME)*

expression: branch
branch:     or         | "if" expression "then" expression "else" expression
or:         and        | and "or" and
and:        not        | not "and" not
not:        comparison | "not" not -> isnot
comparison: arith | arith "==" arith -> eq | arith "!=" arith -> ne
                  | arith ">" arith -> gt  | arith ">=" arith -> ge
                  | arith "<" arith -> lt  | arith "<=" arith -> le
                  | arith "in" table -> in | arith "not" "in" table -> in
arith:   term     | term "+" arith  -> add | term "-" arith      -> sub
term:    factor   | factor "*" term -> mul | factor "/" term     -> div
factor:  pow      | "+" factor      -> pos | "-" factor          -> neg
pow:     call ["**" factor]
call:    atom     | call trailer
atom: "(" expression ")"
    | "{" (sideeffect (NEWLINE | ";"+))* expression "}" -> block
    | CNAME -> symbol
    | NUMBER -> literal
    | ESCAPED_STRING -> string

trailer: "(" arglist ")" -> call
       | "[" arglist "]" -> item
       | "." CNAME -> attr
arglist: expression ("," expression)*

COMMENT: "#" /.*/ NEWLINE | "//" /.*/ NEWLINE | "/*" /(.|\n|\r)*/ "*/"

%import common.CNAME
%import common.NUMBER
%import common.ESCAPED_STRING
%import common.WS
%import common.NEWLINE

%ignore WS
%ignore COMMENT
"""

parser = lark.Lark(grammar)

class AST:
    _fields = ()
    def __init__(self, *args, line=None):
        self.line = line
        for n, x in zip(self._fields, args):
            setattr(self, n, x)
            if self.line is None: self.line = x.line

    def __repr__(self):
        return "{0}({1})".format(type(self).__name__, ", ".join(getattr(self, n) for n in self._fields))

    def __eq__(self, other):
        return type(self) is type(other) and all(getattr(self, n) == getattr(other, n) for n in self._fields)

    def __ne__(self, other):
        return not self.__eq__(other)

class Literal(AST):
    _fields = ("value",)

class Symbol(AST):
    _fields = ("symbol",)

class Block(AST):
    _fields = ("sideeffects", "expression")

class Call(AST):
    _fields = ("function", "arguments")





################################################################################ tests

def test_parser():
    print(parser.parse(r"""
    cut pt > 5 {
        x = 3
        hist eta
    }
    """).pretty())
