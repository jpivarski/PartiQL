import pytest

from awkwardql.interpreter import run
import awkward1 as ak

def test_dataset():
    from awkwardql import data
    events = data.RecordArray({
    "muons": data.ListArray([0, 3, 3, 5], [3, 3, 5, 9], data.RecordArray({
        "pt": data.PrimitiveArray([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]),
        "iso": data.PrimitiveArray([0, 0, 100, 50, 30, 1, 2, 3, 4])
    })),
    "jets": data.ListArray([0, 5, 6, 8], [5, 6, 8, 12], data.RecordArray({
        "pt": data.PrimitiveArray([1, 2, 3, 4, 5, 100, 30, 50, 1, 2, 3, 4]),
        "mass": data.PrimitiveArray([10, 10, 20, 20, 10, 100, 30, 50, 1, 2, 3, 4]),
    })),
    "met": data.PrimitiveArray([100, 200, 300, 400]),
    "stuff": data.ListArray([0, 0, 1, 3], [0, 1, 3, 6], data.PrimitiveArray([1, 2, 2, 3, 3, 3]))
    })
    events.setindex()
    return data.instantiate(events)
    
def test_dataset_awkward():
    data = ak.Array(test_dataset().tolist())
    return data

def test_dataset_realistic():
    from awkwardql import data
    events = data.RecordArray({
    "muons": data.ListArray([0, 3, 3, 5], [3, 3, 5, 9], data.RecordArray({
        "pt": data.PrimitiveArray([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]),
        "charge": data.PrimitiveArray([-1, 1, -1, 1, -1, 1, -1, 1, -1]),
        "iso": data.PrimitiveArray([0, 0, 100, 50, 30, 1, 2, 3, 4])
    })),
    "electrons": data.ListArray([0, 5, 6, 8], [5, 6, 8, 12], data.RecordArray({
        "pt": data.PrimitiveArray([1, 2, 3, 4, 5, 100, 30, 50, 1, 2, 3, 4]),
        "charge": data.PrimitiveArray([1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1]),
        "mass": data.PrimitiveArray([10, 10, 10, 10, 10, 5, 15, 15, 9, 8, 7, 6])
    })),
    "jets": data.ListArray([0, 5, 6, 8], [5, 6, 8, 12], data.RecordArray({
        "pt": data.PrimitiveArray([1, 2, 3, 4, 5, 100, 30, 50, 1, 2, 3, 4]),
        "mass": data.PrimitiveArray([10, 10, 20, 20, 10, 100, 30, 50, 1, 2, 3, 4]),
    })),
    "met": data.PrimitiveArray([100, 200, 300, 400]),
    "stuff": data.ListArray([0, 0, 1, 3], [0, 1, 3, 6], data.PrimitiveArray([1, 2, 2, 3, 3, 3]))
    })
    events.setindex()
    return data.instantiate(events)

def tolist(output):
    if isinstance(output, ak.Array):
        return ak.tolist(output)
    return output.tolist()

@pytest.mark.parametrize("dataset", [test_dataset, test_dataset_awkward])
def test_assign(dataset):
    thedata = dataset()

    output, counter = run(r"""
x = met
""", thedata)
    assert tolist(output) == [{"x": 100}, {"x": 200}, {"x": 300}, {"x": 400}]

    output, counter = run(r"""
x = -met
""", thedata)
    assert tolist(output) == [{"x": -100}, {"x": -200}, {"x": -300}, {"x": -400}]

    output, counter = run(r"""
x = +met
""", thedata)
    assert tolist(output) == [{"x": 100}, {"x": 200}, {"x": 300}, {"x": 400}]

@pytest.mark.parametrize("dataset", [test_dataset, test_dataset_awkward])
def test_scalar(dataset):
    thedata = dataset()

    output, counter = run(r"""
x = met + 1
""", thedata)
    assert tolist(output) == [{"x": 101}, {"x": 201}, {"x": 301}, {"x": 401}]

    output, counter = run(r"""
x = met + 1 + met
""", thedata)
    assert tolist(output) == [{"x": 201}, {"x": 401}, {"x": 601}, {"x": 801}]

    output, counter = run(r"""
x = (met == met)
""", thedata)
    assert tolist(output) == [{"x": True}, {"x": True}, {"x": True}, {"x": True}]

    output, counter = run(r"""
x = (muons == muons)
""", thedata)
    assert tolist(output) == [{"x": True}, {"x": True}, {"x": True}, {"x": True}]

    output, counter = run(r"""
x = (met != met)
""", thedata)
    assert tolist(output) == [{"x": False}, {"x": False}, {"x": False}, {"x": False}]

    output, counter = run(r"""
x = (muons != muons)
""", thedata)
    assert tolist(output) == [{"x": False}, {"x": False}, {"x": False}, {"x": False}]

    output, counter = run(r"""
x = (stuff == stuff)
""", thedata)
    assert tolist(output) == [{"x": True}, {"x": True}, {"x": True}, {"x": True}]

    output, counter = run(r"""
x = (stuff != stuff)
""", thedata)
    assert tolist(output) == [{"x": False}, {"x": False}, {"x": False}, {"x": False}]

    output, counter = run(r"""
x = 1 in stuff
""", thedata)
    assert tolist(output) == [{"x": False}, {"x": True}, {"x": False}, {"x": False}]

    output, counter = run(r"""
x = 2 in stuff
""", thedata)
    assert tolist(output) == [{"x": False}, {"x": False}, {"x": True}, {"x": False}]

    output, counter = run(r"""
x = 1 not in stuff
""", thedata)
    assert tolist(output) == [{"x": True}, {"x": False}, {"x": True}, {"x": True}]

    output, counter = run(r"""
x = 2 not in stuff
""", thedata)
    assert tolist(output) == [{"x": True}, {"x": True}, {"x": False}, {"x": True}]

    output, counter = run(r"""
x = (met == met and met == met)
""", thedata)
    assert tolist(output) == [{"x": True}, {"x": True}, {"x": True}, {"x": True}]

    output, counter = run(r"""
x = (met == met and met != met)
""", thedata)
    assert tolist(output) == [{"x": False}, {"x": False}, {"x": False}, {"x": False}]

    output, counter = run(r"""
x = (met == met or met == met)
""", thedata)
    assert tolist(output) == [{"x": True}, {"x": True}, {"x": True}, {"x": True}]

    output, counter = run(r"""
x = (met != met or met == met)
""", thedata)
    assert tolist(output) == [{"x": True}, {"x": True}, {"x": True}, {"x": True}]

    output, counter = run(r"""
x = (not met == met)
""", thedata)
    assert tolist(output) == [{"x": False}, {"x": False}, {"x": False}, {"x": False}]

    output, counter = run(r"""
x = (not met != met)
""", thedata)
    assert tolist(output) == [{"x": True}, {"x": True}, {"x": True}, {"x": True}]

    output, counter = run(r"""
x = (if 1 in stuff then 1 else -1)
""", thedata)
    assert tolist(output) == [{"x": -1}, {"x": 1}, {"x": -1}, {"x": -1}]

    output, counter = run(r"""
x = {3}
""", thedata)
    assert tolist(output) == [{"x": 3}, {"x": 3}, {"x": 3}, {"x": 3}]

    output, counter = run(r"""
x = { y = 4; y + 2 }
""", thedata)
    assert tolist(output) == [{"x": 6}, {"x": 6}, {"x": 6}, {"x": 6}]

    output, counter = run(r"""
y = 10
x = { y = 4; y + 2 }
""", thedata)
    assert tolist(output) == [{"y": 10, "x": 6}, {"y": 10, "x": 6}, {"y": 10, "x": 6}, {"y": 10, "x": 6}]

#@pytest.mark.parametrize("dataset", [test_dataset, test_dataset_awkward])
def test_scalar_strings():
    thedata = test_dataset()
    
    output, counter = run(r"""
x = (if 2 in stuff then "a" else "b")
""", thedata)
    assert tolist(output) == [{"x": "b"}, {"x": "b"}, {"x": "a"}, {"x": "b"}]

@pytest.mark.parametrize("dataset", [test_dataset, test_dataset_awkward])
def test_tabular_as(dataset):
    thedata = dataset()
    
    output, counter = run(r"""
nested = muons as mu
""", thedata)
    assert tolist(output) == [{"nested": [{"mu": {"pt": 1.1, "iso": 0}}, {"mu": {"pt": 2.2, "iso": 0}}, {"mu": {"pt": 3.3, "iso": 100}}]}, {"nested": []}, {"nested": [{"mu": {"pt": 4.4, "iso": 50}}, {"mu": {"pt": 5.5, "iso": 30}}]}, {"nested": [{"mu": {"pt": 6.6, "iso": 1}}, {"mu": {"pt": 7.7, "iso": 2}}, {"mu": {"pt": 8.8, "iso": 3}}, {"mu": {"pt": 9.9, "iso": 4}}]}]

    output, counter = run(r"""
nested = muons as (m1, m2)
""", thedata)

    assert tolist(output) == [{"nested": [{"m1": {"pt": 1.1, "iso": 0}, "m2": {"pt": 2.2, "iso": 0}}, {"m1": {"pt": 1.1, "iso": 0}, "m2": {"pt": 3.3, "iso": 100}}, {"m1": {"pt": 2.2, "iso": 0}, "m2": {"pt": 3.3, "iso": 100}}]}, {"nested": []}, {"nested": [{"m1": {"pt": 4.4, "iso": 50}, "m2": {"pt": 5.5, "iso": 30}}]}, {"nested": [{"m1": {"pt": 6.6, "iso": 1}, "m2": {"pt": 7.7, "iso": 2}}, {"m1": {"pt": 6.6, "iso": 1}, "m2": {"pt": 8.8, "iso": 3}}, {"m1": {"pt": 6.6, "iso": 1}, "m2": {"pt": 9.9, "iso": 4}}, {"m1": {"pt": 7.7, "iso": 2}, "m2": {"pt": 8.8, "iso": 3}}, {"m1": {"pt": 7.7, "iso": 2}, "m2": {"pt": 9.9, "iso": 4}}, {"m1": {"pt": 8.8, "iso": 3}, "m2": {"pt": 9.9, "iso": 4}}]}]

    output, counter = run(r"""
nested = stuff
""", thedata)
    assert tolist(output) == [{"nested": []}, {"nested": [1]}, {"nested": [2, 2]}, {"nested": [3, 3, 3]}]

    output, counter = run(r"""
nested = stuff as x
""", thedata)
    assert tolist(output) == [{"nested": []}, {"nested": [{"x": 1}]}, {"nested": [{"x": 2}, {"x": 2}]}, {"nested": [{"x": 3}, {"x": 3}, {"x": 3}]}]

@pytest.mark.parametrize("dataset", [test_dataset, test_dataset_awkward])
def test_tabular_with(dataset):
    thedata = dataset()

    output, counter = run(r"""
joined = muons with { iso2 = 2*iso }
""", thedata)
    assert tolist(output) == [{"joined": [{"pt": 1.1, "iso": 0, "iso2": 0}, {"pt": 2.2, "iso": 0, "iso2": 0}, {"pt": 3.3, "iso": 100, "iso2": 200}]}, {"joined": []}, {"joined": [{"pt": 4.4, "iso": 50, "iso2": 100}, {"pt": 5.5, "iso": 30, "iso2": 60}]}, {"joined": [{"pt": 6.6, "iso": 1, "iso2": 2}, {"pt": 7.7, "iso": 2, "iso2": 4}, {"pt": 8.8, "iso": 3, "iso2": 6}, {"pt": 9.9, "iso": 4, "iso2": 8}]}]

    output, counter = run(r"""
joined = muons with { iso2 = 2*iso; iso10 = 10*iso }
""", test_dataset())
    tolist(output) == [{"joined": [{"pt": 1.1, "iso": 0, "iso2": 0, "iso10": 0}, {"pt": 2.2, "iso": 0, "iso2": 0, "iso10": 0}, {"pt": 3.3, "iso": 100, "iso2": 200, "iso10": 1000}]}, {"joined": []}, {"joined": [{"pt": 4.4, "iso": 50, "iso2": 100, "iso10": 500}, {"pt": 5.5, "iso": 30, "iso2": 60, "iso10": 300}]}, {"joined": [{"pt": 6.6, "iso": 1, "iso2": 2, "iso10": 10}, {"pt": 7.7, "iso": 2, "iso2": 4, "iso10": 20}, {"pt": 8.8, "iso": 3, "iso2": 6, "iso10": 30}, {"pt": 9.9, "iso": 4, "iso2": 8, "iso10": 40}]}]

@pytest.mark.parametrize("dataset", [test_dataset, test_dataset_awkward])
def test_tabular_where(dataset):
    thedata = dataset()
    output, counter = run(r"""
joined = muons where iso > 2
""", thedata)
    assert tolist(output) == [{"joined": [{"pt": 3.3, "iso": 100}]}, {"joined": []}, {"joined": [{"pt": 4.4, "iso": 50}, {"pt": 5.5, "iso": 30}]}, {"joined": [{"pt": 8.8, "iso": 3}, {"pt": 9.9, "iso": 4}]}]
    
    output, counter = run(r"""
joined = muons where pt < 5 or iso > 2
""", thedata)
    assert tolist(output) == [{"joined": [{"pt": 1.1, "iso": 0}, {"pt": 2.2, "iso": 0}, {"pt": 3.3, "iso": 100}]}, {"joined": []}, {"joined": [{"pt": 4.4, "iso": 50}, {"pt": 5.5, "iso": 30}]}, {"joined": [{"pt": 8.8, "iso": 3}, {"pt": 9.9, "iso": 4}]}]
    
    output, counter = run(r"""
joined = muons where iso > 2 with { iso2 = 2*iso }
""", test_dataset())
    assert tolist(output) == [{"joined": [{"pt": 3.3, "iso": 100, "iso2": 200}]}, {"joined": []}, {"joined": [{"pt": 4.4, "iso": 50, "iso2": 100}, {"pt": 5.5, "iso": 30, "iso2": 60}]}, {"joined": [{"pt": 8.8, "iso": 3, "iso2": 6}, {"pt": 9.9, "iso": 4, "iso2": 8}]}]

@pytest.mark.parametrize("dataset", [test_dataset, test_dataset_awkward])
def test_tabular_where_union(dataset):
    thedata = dataset()

    output, counter = run(r"""
joined = muons where iso > 2 union muons
""", thedata)
    assert tolist(output) == [{"joined": [{"pt": 3.3, "iso": 100}, {"pt": 1.1, "iso": 0}, {"pt": 2.2, "iso": 0}]}, {"joined": []}, {"joined": [{"pt": 4.4, "iso": 50}, {"pt": 5.5, "iso": 30}]}, {"joined": [{"pt": 8.8, "iso": 3}, {"pt": 9.9, "iso": 4}, {"pt": 6.6, "iso": 1}, {"pt": 7.7, "iso": 2}]}]

    thedata = test_dataset()
    output, counter = run(r"""
joined = muons where iso > 2 union muons where pt < 5
""", thedata)
    assert tolist(output) == [{"joined": [{"pt": 3.3, "iso": 100}, {"pt": 1.1, "iso": 0}, {"pt": 2.2, "iso": 0}]}, {"joined": []}, {"joined": [{"pt": 4.4, "iso": 50}, {"pt": 5.5, "iso": 30}]}, {"joined": [{"pt": 8.8, "iso": 3}, {"pt": 9.9, "iso": 4}]}]

    output, counter = run(r"""
joined = muons where pt < 5 union muons where iso > 2
""", thedata)
    assert tolist(output) == [{"joined": [{"pt": 1.1, "iso": 0}, {"pt": 2.2, "iso": 0}, {"pt": 3.3, "iso": 100}]}, {"joined": []}, {"joined": [{"pt": 4.4, "iso": 50}, {"pt": 5.5, "iso": 30}]}, {"joined": [{"pt": 8.8, "iso": 3}, {"pt": 9.9, "iso": 4}]}]


def test_tabular_where_with_union():
    output, counter = run(r"""
joined = muons where iso > 2 with { iso2 = 2*iso } union muons
""", test_dataset())
    assert tolist(output) == [{"joined": [{"pt": 3.3, "iso": 100, "iso2": 200}, {"pt": 1.1, "iso": 0}, {"pt": 2.2, "iso": 0}]}, {"joined": []}, {"joined": [{"pt": 4.4, "iso": 50, "iso2": 100}, {"pt": 5.5, "iso": 30, "iso2": 60}]}, {"joined": [{"pt": 8.8, "iso": 3, "iso2": 6}, {"pt": 9.9, "iso": 4, "iso2": 8}, {"pt": 6.6, "iso": 1}, {"pt": 7.7, "iso": 2}]}]

    output, counter = run(r"""
joined = muons where iso > 2 with { iso2 = 2*iso } union muons where pt < 5
""", test_dataset())
    assert tolist(output) == [{"joined": [{"pt": 3.3, "iso": 100, "iso2": 200}, {"pt": 1.1, "iso": 0}, {"pt": 2.2, "iso": 0}]}, {"joined": []}, {"joined": [{"pt": 4.4, "iso": 50, "iso2": 100}, {"pt": 5.5, "iso": 30, "iso2": 60}]}, {"joined": [{"pt": 8.8, "iso": 3, "iso2": 6}, {"pt": 9.9, "iso": 4, "iso2": 8}]}]

@pytest.mark.parametrize("dataset", [test_dataset, test_dataset_awkward])
def test_tabular_cross(dataset):
    thedata = dataset()

    output, counter = run(r"""
joined = muons cross jets
""", thedata)
    print(tolist(output))
    assert tolist(output) == [{"joined": [{"pt": 1.1, "iso": 0, "mass": 10}, {"pt": 1.1, "iso": 0, "mass": 10}, {"pt": 1.1, "iso": 0, "mass": 20}, {"pt": 1.1, "iso": 0, "mass": 20}, {"pt": 1.1, "iso": 0, "mass": 10}, {"pt": 2.2, "iso": 0, "mass": 10}, {"pt": 2.2, "iso": 0, "mass": 10}, {"pt": 2.2, "iso": 0, "mass": 20}, {"pt": 2.2, "iso": 0, "mass": 20}, {"pt": 2.2, "iso": 0, "mass": 10}, {"pt": 3.3, "iso": 100, "mass": 10}, {"pt": 3.3, "iso": 100, "mass": 10}, {"pt": 3.3, "iso": 100, "mass": 20}, {"pt": 3.3, "iso": 100, "mass": 20}, {"pt": 3.3, "iso": 100, "mass": 10}]}, {"joined": []}, {"joined": [{"pt": 4.4, "iso": 50, "mass": 30}, {"pt": 4.4, "iso": 50, "mass": 50}, {"pt": 5.5, "iso": 30, "mass": 30}, {"pt": 5.5, "iso": 30, "mass": 50}]}, {"joined": [{"pt": 6.6, "iso": 1, "mass": 1}, {"pt": 6.6, "iso": 1, "mass": 2}, {"pt": 6.6, "iso": 1, "mass": 3}, {"pt": 6.6, "iso": 1, "mass": 4}, {"pt": 7.7, "iso": 2, "mass": 1}, {"pt": 7.7, "iso": 2, "mass": 2}, {"pt": 7.7, "iso": 2, "mass": 3}, {"pt": 7.7, "iso": 2, "mass": 4}, {"pt": 8.8, "iso": 3, "mass": 1}, {"pt": 8.8, "iso": 3, "mass": 2}, {"pt": 8.8, "iso": 3, "mass": 3}, {"pt": 8.8, "iso": 3, "mass": 4}, {"pt": 9.9, "iso": 4, "mass": 1}, {"pt": 9.9, "iso": 4, "mass": 2}, {"pt": 9.9, "iso": 4, "mass": 3}, {"pt": 9.9, "iso": 4, "mass": 4}]}]

@pytest.mark.parametrize("dataset", [test_dataset, test_dataset_awkward])
def test_tabular_join(dataset):
    thedata = dataset()

    output, counter = run(r"""
whatever = muons join jets
""", thedata)
    assert tolist(output) == [{"whatever": []}, {"whatever": []}, {"whatever": []}, {"whatever": []}]

    output, counter = run(r"""
joined = muons where iso > 2 join muons where pt < 5
""", thedata)
    assert tolist(output) == [{"joined": [{"pt": 3.3, "iso": 100}]}, {"joined": []}, {"joined": [{"pt": 4.4, "iso": 50}]}, {"joined": []}]

    output, counter = run(r"""
joined = muons where pt < 5 join muons where iso > 2
""", thedata)
    assert tolist(output) == [{"joined": [{"pt": 3.3, "iso": 100}]}, {"joined": []}, {"joined": [{"pt": 4.4, "iso": 50}]}, {"joined": []}]

    output, counter = run(r"""
joined = muons where pt < 5 and iso > 2
""", thedata)
    assert tolist(output) == [{"joined": [{"pt": 3.3, "iso": 100}]}, {"joined": []}, {"joined": [{"pt": 4.4, "iso": 50}]}, {"joined": []}]

def test_tabular_join_with():

    output, counter = run(r"""
joined = muons where iso > 2 with { iso2 = 2*iso } join muons
""", test_dataset())
    assert tolist(output) == [{"joined": [{"pt": 3.3, "iso": 100, "iso2": 200}]}, {"joined": []}, {"joined": [{"pt": 4.4, "iso": 50, "iso2": 100}, {"pt": 5.5, "iso": 30, "iso2": 60}]}, {"joined": [{"pt": 8.8, "iso": 3, "iso2": 6}, {"pt": 9.9, "iso": 4, "iso2": 8}]}]

@pytest.mark.parametrize("dataset", [test_dataset, test_dataset_awkward])
def test_tabular_join_except(dataset):
    thedata = dataset()
    
    output, counter = run(r"""
joined = muons where pt < 7 except muons where iso > 2
""", thedata)
    assert tolist(output) == [{"joined": [{"pt": 1.1, "iso": 0}, {"pt": 2.2, "iso": 0}]}, {"joined": []}, {"joined": []}, {"joined": [{"pt": 6.6, "iso": 1}]}]

    output, counter = run(r"""
joined = muons where pt < 7 and not iso > 2
""", thedata)
    assert tolist(output) == [{"joined": [{"pt": 1.1, "iso": 0}, {"pt": 2.2, "iso": 0}]}, {"joined": []}, {"joined": []}, {"joined": [{"pt": 6.6, "iso": 1}]}]

def test_tabular_group():
    output, counter = run(r"""
grouped = jets group by mass
""", test_dataset())
    assert output.tolist() == [{"grouped": [[{"pt": 1, "mass": 10}, {"pt": 2, "mass": 10}, {"pt": 5, "mass": 10}], [{"pt": 3, "mass": 20}, {"pt": 4, "mass": 20}]]}, {"grouped": [[{"pt": 100, "mass": 100}]]}, {"grouped": [[{"pt": 30, "mass": 30}], [{"pt": 50, "mass": 50}]]}, {"grouped": [[{"pt": 1, "mass": 1}], [{"pt": 2, "mass": 2}], [{"pt": 3, "mass": 3}], [{"pt": 4, "mass": 4}]]}]

    output, counter = run(r"""
grouped = muons as m1 cross muons as m2 group by m1
""", test_dataset())
    assert output.tolist() == [{"grouped": [[{"m1": {"pt": 1.1, "iso": 0}, "m2": {"pt": 1.1, "iso": 0}}, {"m1": {"pt": 1.1, "iso": 0}, "m2": {"pt": 2.2, "iso": 0}}, {"m1": {"pt": 1.1, "iso": 0}, "m2": {"pt": 3.3, "iso": 100}}], [{"m1": {"pt": 2.2, "iso": 0}, "m2": {"pt": 1.1, "iso": 0}}, {"m1": {"pt": 2.2, "iso": 0}, "m2": {"pt": 2.2, "iso": 0}}, {"m1": {"pt": 2.2, "iso": 0}, "m2": {"pt": 3.3, "iso": 100}}], [{"m1": {"pt": 3.3, "iso": 100}, "m2": {"pt": 1.1, "iso": 0}}, {"m1": {"pt": 3.3, "iso": 100}, "m2": {"pt": 2.2, "iso": 0}}, {"m1": {"pt": 3.3, "iso": 100}, "m2": {"pt": 3.3, "iso": 100}}]]}, {"grouped": []}, {"grouped": [[{"m1": {"pt": 4.4, "iso": 50}, "m2": {"pt": 4.4, "iso": 50}}, {"m1": {"pt": 4.4, "iso": 50}, "m2": {"pt": 5.5, "iso": 30}}], [{"m1": {"pt": 5.5, "iso": 30}, "m2": {"pt": 4.4, "iso": 50}}, {"m1": {"pt": 5.5, "iso": 30}, "m2": {"pt": 5.5, "iso": 30}}]]}, {"grouped": [[{"m1": {"pt": 6.6, "iso": 1}, "m2": {"pt": 6.6, "iso": 1}}, {"m1": {"pt": 6.6, "iso": 1}, "m2": {"pt": 7.7, "iso": 2}}, {"m1": {"pt": 6.6, "iso": 1}, "m2": {"pt": 8.8, "iso": 3}}, {"m1": {"pt": 6.6, "iso": 1}, "m2": {"pt": 9.9, "iso": 4}}], [{"m1": {"pt": 7.7, "iso": 2}, "m2": {"pt": 6.6, "iso": 1}}, {"m1": {"pt": 7.7, "iso": 2}, "m2": {"pt": 7.7, "iso": 2}}, {"m1": {"pt": 7.7, "iso": 2}, "m2": {"pt": 8.8, "iso": 3}}, {"m1": {"pt": 7.7, "iso": 2}, "m2": {"pt": 9.9, "iso": 4}}], [{"m1": {"pt": 8.8, "iso": 3}, "m2": {"pt": 6.6, "iso": 1}}, {"m1": {"pt": 8.8, "iso": 3}, "m2": {"pt": 7.7, "iso": 2}}, {"m1": {"pt": 8.8, "iso": 3}, "m2": {"pt": 8.8, "iso": 3}}, {"m1": {"pt": 8.8, "iso": 3}, "m2": {"pt": 9.9, "iso": 4}}], [{"m1": {"pt": 9.9, "iso": 4}, "m2": {"pt": 6.6, "iso": 1}}, {"m1": {"pt": 9.9, "iso": 4}, "m2": {"pt": 7.7, "iso": 2}}, {"m1": {"pt": 9.9, "iso": 4}, "m2": {"pt": 8.8, "iso": 3}}, {"m1": {"pt": 9.9, "iso": 4}, "m2": {"pt": 9.9, "iso": 4}}]]}]

    output, counter = run(r"""
grouped = muons as m1 cross muons as m2 group by m2
""", test_dataset())
    assert output.tolist() == [{"grouped": [[{"m1": {"pt": 1.1, "iso": 0}, "m2": {"pt": 1.1, "iso": 0}}, {"m1": {"pt": 2.2, "iso": 0}, "m2": {"pt": 1.1, "iso": 0}}, {"m1": {"pt": 3.3, "iso": 100}, "m2": {"pt": 1.1, "iso": 0}}], [{"m1": {"pt": 1.1, "iso": 0}, "m2": {"pt": 2.2, "iso": 0}}, {"m1": {"pt": 2.2, "iso": 0}, "m2": {"pt": 2.2, "iso": 0}}, {"m1": {"pt": 3.3, "iso": 100}, "m2": {"pt": 2.2, "iso": 0}}], [{"m1": {"pt": 1.1, "iso": 0}, "m2": {"pt": 3.3, "iso": 100}}, {"m1": {"pt": 2.2, "iso": 0}, "m2": {"pt": 3.3, "iso": 100}}, {"m1": {"pt": 3.3, "iso": 100}, "m2": {"pt": 3.3, "iso": 100}}]]}, {"grouped": []}, {"grouped": [[{"m1": {"pt": 4.4, "iso": 50}, "m2": {"pt": 4.4, "iso": 50}}, {"m1": {"pt": 5.5, "iso": 30}, "m2": {"pt": 4.4, "iso": 50}}], [{"m1": {"pt": 4.4, "iso": 50}, "m2": {"pt": 5.5, "iso": 30}}, {"m1": {"pt": 5.5, "iso": 30}, "m2": {"pt": 5.5, "iso": 30}}]]}, {"grouped": [[{"m1": {"pt": 6.6, "iso": 1}, "m2": {"pt": 6.6, "iso": 1}}, {"m1": {"pt": 7.7, "iso": 2}, "m2": {"pt": 6.6, "iso": 1}}, {"m1": {"pt": 8.8, "iso": 3}, "m2": {"pt": 6.6, "iso": 1}}, {"m1": {"pt": 9.9, "iso": 4}, "m2": {"pt": 6.6, "iso": 1}}], [{"m1": {"pt": 6.6, "iso": 1}, "m2": {"pt": 7.7, "iso": 2}}, {"m1": {"pt": 7.7, "iso": 2}, "m2": {"pt": 7.7, "iso": 2}}, {"m1": {"pt": 8.8, "iso": 3}, "m2": {"pt": 7.7, "iso": 2}}, {"m1": {"pt": 9.9, "iso": 4}, "m2": {"pt": 7.7, "iso": 2}}], [{"m1": {"pt": 6.6, "iso": 1}, "m2": {"pt": 8.8, "iso": 3}}, {"m1": {"pt": 7.7, "iso": 2}, "m2": {"pt": 8.8, "iso": 3}}, {"m1": {"pt": 8.8, "iso": 3}, "m2": {"pt": 8.8, "iso": 3}}, {"m1": {"pt": 9.9, "iso": 4}, "m2": {"pt": 8.8, "iso": 3}}], [{"m1": {"pt": 6.6, "iso": 1}, "m2": {"pt": 9.9, "iso": 4}}, {"m1": {"pt": 7.7, "iso": 2}, "m2": {"pt": 9.9, "iso": 4}}, {"m1": {"pt": 8.8, "iso": 3}, "m2": {"pt": 9.9, "iso": 4}}, {"m1": {"pt": 9.9, "iso": 4}, "m2": {"pt": 9.9, "iso": 4}}]]}]

    output, counter = run(r"""
x = 3
extreme = muons max by pt
""", test_dataset())
    assert output.tolist() == [{"x": 3, "extreme": {"pt": 3.3, "iso": 100}}, {"x": 3}, {"x": 3, "extreme": {"pt": 5.5, "iso": 30}}, {"x": 3, "extreme": {"pt": 9.9, "iso": 4}}]

    output, counter = run(r"""
x = 3
extreme = muons min by pt
""", test_dataset())
    assert output.tolist() == [{"x": 3, "extreme": {"pt": 1.1, "iso": 0}}, {"x": 3}, {"x": 3, "extreme": {"pt": 4.4, "iso": 50}}, {"x": 3, "extreme": {"pt": 6.6, "iso": 1}}]

    output, counter = run(r"""
extreme = muons min by pt
whatever = has extreme
""", test_dataset())
    assert output.tolist() == [{"extreme": {"pt": 1.1, "iso": 0}, "whatever": True}, {"whatever": False}, {"extreme": {"pt": 4.4, "iso": 50}, "whatever": True}, {"extreme": {"pt": 6.6, "iso": 1}, "whatever": True}]

    output, counter = run(r"""
extreme = muons min by pt
whatever = if has extreme then extreme.iso else -100
""", test_dataset())
    assert output.tolist() == [{"extreme": {"pt": 1.1, "iso": 0}, "whatever": 0}, {"whatever": -100}, {"extreme": {"pt": 4.4, "iso": 50}, "whatever": 50}, {"extreme": {"pt": 6.6, "iso": 1}, "whatever": 1}]

    output, counter = run(r"""
x = 3
extreme = muons min by pt
whatever = if has extreme then extreme.iso
""", test_dataset())
    assert output.tolist() == [{"x": 3, "extreme": {"pt": 1.1, "iso": 0}, "whatever": 0}, {"x": 3}, {"x": 3, "extreme": {"pt": 4.4, "iso": 50}, "whatever": 50}, {"x": 3, "extreme": {"pt": 6.6, "iso": 1}, "whatever": 1}]

    output, counter = run(r"""
extreme = muons where iso > 2 with { iso2 = 2*iso } union muons min by pt
whatever = if has extreme then extreme?.iso2
""", test_dataset())
    assert output.tolist() == [{"extreme": {"pt": 1.1, "iso": 0}}, {"extreme": {"pt": 4.4, "iso": 50, "iso2": 100}, "whatever": 100}, {"extreme": {"pt": 6.6, "iso": 1}}]

    output, counter = run(r"""
extreme = muons where iso > 2 with { iso2 = 2*iso } union muons min by pt
whatever = ?extreme?.iso2
""", test_dataset())
    assert output.tolist() == [{"extreme": {"pt": 1.1, "iso": 0}}, {"extreme": {"pt": 4.4, "iso": 50, "iso2": 100}, "whatever": 100}, {"extreme": {"pt": 6.6, "iso": 1}}]

def test_realistic_queries():
    output, counter = run(r"""
def mass(one, two) {
    91.2 + one.pt + two.pt # yes this is just a stand in
}
best_leptons = {    
    Z = electrons as (lep1, lep2) union muons as (lep1, lep2) 
               where lep1.charge != lep2.charge               
               min by abs(mass(lep1, lep2) - 91.2)
    [?Z.lep1, ?Z.lep2]
}
leading = best_leptons max by pt
trailing = best_leptons min by pt
""", test_dataset_realistic())
    assert output.tolist() == [{'best_leptons': [{'pt': 1, 'charge': 1, 'mass': 10},{'pt': 2, 'charge': -1, 'mass': 10}],'leading': {'pt': 2, 'charge': -1, 'mass': 10},'trailing': {'pt': 1, 'charge': 1, 'mass': 10}},{'best_leptons': []},{'best_leptons': [{'pt': 4.4, 'charge': 1, 'iso': 50},{'pt': 5.5, 'charge': -1, 'iso': 30}],'leading': {'pt': 5.5, 'charge': -1, 'iso': 30},'trailing': {'pt': 4.4, 'charge': 1, 'iso': 50}},{'best_leptons': [{'pt': 1, 'charge': 1, 'mass': 9},{'pt': 2, 'charge': -1, 'mass': 8}],'leading': {'pt': 2, 'charge': -1, 'mass': 8},'trailing': {'pt': 1, 'charge': 1, 'mass': 9}}]

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

    output, counter = run(r"""
hist met by regular(10, 0, 1000)
""", test_dataset())
    counts, edges = counter["0"].numpy()
    assert counts.tolist() == [0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    assert edges.tolist() == [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    output, counter = run(r"""
hist met by regular(10, 0, 1000), met - 1 by regular(5, 0, 500)
""", test_dataset())
    counts, (xedges, yedges) = counter["0"].numpy()
    assert counts.tolist() == [[0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]
    assert xedges.tolist() == [0.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0]
    assert yedges.tolist() == [0.0, 100.0, 200.0, 300.0, 400.0, 500.0]

    output, counter = run(r"""
hist met by regular(10, 0, 1000) weight by 2
""", test_dataset())
    counts, edges = counter["0"].numpy()
    assert counts.tolist() == [0, 2, 2, 2, 2, 0, 0, 0, 0, 0]
    assert edges.tolist() == [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    output, counter = run(r"""
hist met by regular(10, 0, 1000) named "one" titled "two"
""", test_dataset())
    assert counter["one"].title == "two"

def test_cutvary():
    output, counter = run(r"""
vary by { x = 1 } by { x = 2 } by { x = 3 } {
    hist x by regular(5, 0, 5)
}
""", test_dataset())
    assert counter.allkeys() == ["0", "0/0", "1", "1/0", "2", "2/0"]
    assert counter["0/0"].numpy()[0].tolist() == [0, 4, 0, 0, 0]
    assert counter["1/0"].numpy()[0].tolist() == [0, 0, 4, 0, 0]
    assert counter["2/0"].numpy()[0].tolist() == [0, 0, 0, 4, 0]

    output, counter = run(r"""
vary by { x = 1 } named "one" by { x = 2 } named "two" by { x = 3 } named "three" {
    hist x by regular(5, 0, 5)
}
""", test_dataset())
    assert counter.allkeys() == ["one", "one/0", "two", "two/0", "three", "three/0"]
    assert counter["one/0"].numpy()[0].tolist() == [0, 4, 0, 0, 0]
    assert counter["two/0"].numpy()[0].tolist() == [0, 0, 4, 0, 0]
    assert counter["three/0"].numpy()[0].tolist() == [0, 0, 0, 4, 0]

    output, counter = run(r"""
cut met > 200 {
    hist met by regular(5, 0, 500)
}
""", test_dataset())
    assert counter.allkeys() == ["0", "0/0"]
    assert counter["0/0"].numpy()[0].tolist() == [0, 0, 0, 1, 1]

    output, counter = run(r"""
cut met > 200 weight by 2 {
    hist met by regular(5, 0, 500)
}
""", test_dataset())
    assert counter.allkeys() == ["0", "0/0"]
    assert counter["0/0"].numpy()[0].tolist() == [0, 0, 0, 2, 2]

    output, counter = run(r"""
cut met > 200 named "one" titled "two" {
    hist met by regular(5, 0, 500)
}
""", test_dataset())
    assert counter.allkeys() == ["one", "one/0"]
    assert counter["one/0"].numpy()[0].tolist() == [0, 0, 0, 1, 1]
    assert counter["one"].title == "two"

