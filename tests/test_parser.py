import pytest

from awkwardql.parser import (parse, Symbol, Literal, Assignment,
                              Pack, Histogram, Cut, Call, With, Axis,
                              GetAttr, Block, Vary, Trial)

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
    assert parse(r"?x") == [Symbol("x", True)]
    assert parse(r"1") == [Literal(1)]
    assert parse(r"3.14") == [Literal(3.14)]
    assert parse(r'"hello"') == [Literal("hello")]
    assert parse(r"f(x)") == [Call(Symbol("f"), [Symbol("x")])]
    assert parse(r"f(x, 1, 3.14)") == [Call(Symbol("f"), [Symbol("x"), Literal(1), Literal(3.14)])]
    # parse(r"a[0]")
    # assert parse(r"a[0]") == [GetItem(Symbol("a"), [Literal(0)])]
    # assert parse(r"a[0][i]") == [GetItem(GetItem(Symbol("a"), [Literal(0)]), [Symbol("i")])]
    # assert parse(r"a[0, i]") == [GetItem(Symbol("a"), [Literal(0), Symbol("i")])]
    assert parse(r"a.b") == [GetAttr(Symbol("a"), "b", False)]
    assert parse(r"a.b.c") == [GetAttr(GetAttr(Symbol("a"), "b", False), "c", False)]
    assert parse(r"a?.b") == [GetAttr(Symbol("a"), "b", True)]
    assert parse(r"a?.b?.c") == [GetAttr(GetAttr(Symbol("a"), "b", True), "c", True)]
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
    assert parse(r"x in table") == [Call(Symbol(".in"), [Symbol("x"), Symbol("table")])]
    assert parse(r"x not in table") == [Call(Symbol(".not in"), [Symbol("x"), Symbol("table")])]
    assert parse(r"p and q") == [Call(Symbol(".and"), [Symbol("p"), Symbol("q")])]
    assert parse(r"p or q") == [Call(Symbol(".or"), [Symbol("p"), Symbol("q")])]
    assert parse(r"not p") == [Call(Symbol(".not"), [Symbol("p")])]
    assert parse(r"p or q and r") == [Call(Symbol(".or"), [Symbol("p"), Call(Symbol(".and"), [Symbol("q"), Symbol("r")])])]
    assert parse(r"(p or q) and r") == [Call(Symbol(".and"), [Call(Symbol(".or"), [Symbol("p"), Symbol("q")]), Symbol("r")])]
    assert parse(r"if x > 0 then 1 else -1") == [Call(Symbol(".if"), [Call(Symbol(">"), [Symbol("x"), Literal(0)]), Literal(1), Call(Symbol("*-1"), [Literal(1)])])]
    assert parse(r"if p then if q then 1 else 2 else 3") == [Call(Symbol(".if"), [Symbol("p"), Call(Symbol(".if"), [Symbol("q"), Literal(1), Literal(2)]), Literal(3)])]
    assert parse(r"if p then { if q then 1 else 2 } else 3") == [Call(Symbol(".if"), [Symbol("p"), Block([Call(Symbol(".if"), [Symbol("q"), Literal(1), Literal(2)])]), Literal(3)])]
    assert parse(r"if p then 1 else if q then 2 else 3") == [Call(Symbol(".if"), [Symbol("p"), Literal(1), Call(Symbol(".if"), [Symbol("q"), Literal(2), Literal(3)])])]
    assert parse(r"if p then 1 else { if q then 2 else 3 }") == [Call(Symbol(".if"), [Symbol("p"), Literal(1), Block([Call(Symbol(".if"), [Symbol("q"), Literal(2), Literal(3)])])])]

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
    assert parse(r"if x > 0 then {1} else {-1}") == [Call(Symbol(".if"), [Call(Symbol(">"), [Symbol("x"), Literal(0)]), Block([Literal(1)]), Block([Call(Symbol("*-1"), [Literal(1)])])])]

def test_table():
    assert parse(r"table as x") == [Pack(Symbol("table"), ["x"])]
    assert parse(r"table as (x, y)") == [Pack(Symbol("table"), ["x", "y"])]
    assert parse(r"table with { x = 3 }") == [With(Symbol("table"), [Assignment("x", Literal(3))], False)]
    assert parse(r"table with { x = 3; y = x }") == [With(Symbol("table"), [Assignment("x", Literal(3)), Assignment("y", Symbol("x"))], False)]
    assert parse(r"table where x > 0") == [Call(Symbol(".where"), [Symbol("table"), Call(Symbol(">"), [Symbol("x"), Literal(0)])])]
    assert parse(r"table with { x = 3 } where x > 0") == [Call(Symbol(".where"), [With(Symbol("table"), [Assignment("x", Literal(3))], False), Call(Symbol(">"), [Symbol("x"), Literal(0)])])]
    assert parse(r"a join b") == [Call(Symbol(".join"), [Symbol("a"), Symbol("b")])]
    assert parse(r"a cross b") == [Call(Symbol(".cross"), [Symbol("a"), Symbol("b")])]
    assert parse(r"a cross b join c") == [Call(Symbol(".join"), [Call(Symbol(".cross"), [Symbol("a"), Symbol("b")]), Symbol("c")])]
    assert parse(r"(a cross b) join c") == [Call(Symbol(".join"), [Call(Symbol(".cross"), [Symbol("a"), Symbol("b")]), Symbol("c")])]
    assert parse(r"a except b union c") == [Call(Symbol(".union"), [Call(Symbol(".except"), [Symbol("a"), Symbol("b")]), Symbol("c")])]
    assert parse(r"(a except b) union c") == [Call(Symbol(".union"), [Call(Symbol(".except"), [Symbol("a"), Symbol("b")]), Symbol("c")])]
    assert parse(r"a union b cross c") == [Call(Symbol(".union"), [Symbol("a"), Call(Symbol(".cross"), [Symbol("b"), Symbol("c")])])]
    assert parse(r"(a union b) cross c") == [Call(Symbol(".cross"), [Call(Symbol(".union"), [Symbol("a"), Symbol("b")]), Symbol("c")])]
    assert parse(r"a union b join c") == [Call(Symbol(".union"), [Symbol("a"), Call(Symbol(".join"), [Symbol("b"), Symbol("c")])])]
    assert parse(r"(a union b) join c") == [Call(Symbol(".join"), [Call(Symbol(".union"), [Symbol("a"), Symbol("b")]), Symbol("c")])]
    assert parse(r"a except b cross c") == [Call(Symbol(".except"), [Symbol("a"), Call(Symbol(".cross"), [Symbol("b"), Symbol("c")])])]
    assert parse(r"(a except b) cross c") == [Call(Symbol(".cross"), [Call(Symbol(".except"), [Symbol("a"), Symbol("b")]), Symbol("c")])]
    assert parse(r"a except b join c") == [Call(Symbol(".except"), [Symbol("a"), Call(Symbol(".join"), [Symbol("b"), Symbol("c")])])]
    assert parse(r"(a except b) join c") == [Call(Symbol(".join"), [Call(Symbol(".except"), [Symbol("a"), Symbol("b")]), Symbol("c")])]
    assert parse(r"a join b join c") == [Call(Symbol(".join"), [Call(Symbol(".join"), [Symbol("a"), Symbol("b")]), Symbol("c")])]
    assert parse(r"a cross b cross c") == [Call(Symbol(".cross"), [Call(Symbol(".cross"), [Symbol("a"), Symbol("b")]), Symbol("c")])]
    assert parse(r"a union b union c") == [Call(Symbol(".union"), [Call(Symbol(".union"), [Symbol("a"), Symbol("b")]), Symbol("c")])]
    assert parse(r"a except b except c") == [Call(Symbol(".except"), [Call(Symbol(".except"), [Symbol("a"), Symbol("b")]), Symbol("c")])]
    assert parse(r"table group by x") == [Call(Symbol(".group"), [Symbol("table"), Symbol("x")])]
    assert parse(r"(table group by x) with { y = 4 }") == [With(Call(Symbol(".group"), [Symbol("table"), Symbol("x")]), [Assignment("y", Literal(4))], False)]
    assert parse(r"table min by x") == [Call(Symbol(".min"), [Symbol("table"), Symbol("x")])]
    assert parse(r"table max by x") == [Call(Symbol(".max"), [Symbol("table"), Symbol("x")])]

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
    """For events with at least three leptons and a same-flavor
       opposite-sign lepton pair, find the same-flavor opposite-sign
       lepton pair with the mass closest to 91.2 GeV and plot the pT
       of the leading other lepton."""
    assert parse(r"""
leptons = electrons union muons

cut count(leptons) >= 3 named "three_leptons" {
    Z = electrons as (lep1, lep2) union muons as (lep1, lep2)
            where lep1.charge != lep2.charge
            min by abs(mass(lep1, lep2) - 91.2)

    third = leptons except [Z.lep1, Z.lep2] max by pt

    hist third.pt by regular(100, 0, 250) named "third_pt" titled "Leading other lepton pT"
}
""")

