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
        "iso": data.PrimitiveArray([10, 10, 10, 10, 10, 5, 15, 15, 9, 8, 7, 6])
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

def test_dataset_realistic_awkward():
    data = ak.Array(test_dataset_realistic().tolist())
    return data

def removeNones(d):
    clean = None
    print(type(d),':',d)
    if isinstance(d, list):
        clean = [removeNones(v) for v in d]
    elif isinstance(d, dict):
        clean = {}
        for k, v in d.items():
            if isinstance(v, dict):
                nested = removeNones(v)
                if len(nested.keys()) > 0:
                    clean[k] = nested
            elif isinstance(v, list):
                cleaned = []
                for val in v:
                    if val is None:
                        continue
                    if isinstance(val, (dict, list)):
                        cleaned.append(removeNones(val))
                    else:
                        cleaned.append(val)
                clean[k] = cleaned
                
            elif v is not None:
                clean[k] = v
    else:
        raise Exception('Nope')
    return clean

def tolist(output):
    if isinstance(output, ak.Array):
        return removeNones(ak.tolist(output))
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

@pytest.mark.parametrize("dataset", [test_dataset, test_dataset_awkward])
def test_scalar_strings(dataset):
    thedata = dataset()
    
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

@pytest.mark.parametrize("dataset", [test_dataset, test_dataset_awkward])
def test_tabular_group(dataset):
    thedata = dataset()
    
    output, counter = run(r"""
grouped = jets group by mass
""", thedata)
    assert tolist(output) == [{"grouped": [[{"pt": 1, "mass": 10}, {"pt": 2, "mass": 10}, {"pt": 5, "mass": 10}], [{"pt": 3, "mass": 20}, {"pt": 4, "mass": 20}]]}, {"grouped": [[{"pt": 100, "mass": 100}]]}, {"grouped": [[{"pt": 30, "mass": 30}], [{"pt": 50, "mass": 50}]]}, {"grouped": [[{"pt": 1, "mass": 1}], [{"pt": 2, "mass": 2}], [{"pt": 3, "mass": 3}], [{"pt": 4, "mass": 4}]]}]

    output, counter = run(r"""
grouped = muons as m1 cross muons as m2 group by m1
""", thedata)
    assert tolist(output) == [{"grouped": [[{"m1": {"pt": 1.1, "iso": 0}, "m2": {"pt": 1.1, "iso": 0}}, {"m1": {"pt": 1.1, "iso": 0}, "m2": {"pt": 2.2, "iso": 0}}, {"m1": {"pt": 1.1, "iso": 0}, "m2": {"pt": 3.3, "iso": 100}}], [{"m1": {"pt": 2.2, "iso": 0}, "m2": {"pt": 1.1, "iso": 0}}, {"m1": {"pt": 2.2, "iso": 0}, "m2": {"pt": 2.2, "iso": 0}}, {"m1": {"pt": 2.2, "iso": 0}, "m2": {"pt": 3.3, "iso": 100}}], [{"m1": {"pt": 3.3, "iso": 100}, "m2": {"pt": 1.1, "iso": 0}}, {"m1": {"pt": 3.3, "iso": 100}, "m2": {"pt": 2.2, "iso": 0}}, {"m1": {"pt": 3.3, "iso": 100}, "m2": {"pt": 3.3, "iso": 100}}]]}, {"grouped": []}, {"grouped": [[{"m1": {"pt": 4.4, "iso": 50}, "m2": {"pt": 4.4, "iso": 50}}, {"m1": {"pt": 4.4, "iso": 50}, "m2": {"pt": 5.5, "iso": 30}}], [{"m1": {"pt": 5.5, "iso": 30}, "m2": {"pt": 4.4, "iso": 50}}, {"m1": {"pt": 5.5, "iso": 30}, "m2": {"pt": 5.5, "iso": 30}}]]}, {"grouped": [[{"m1": {"pt": 6.6, "iso": 1}, "m2": {"pt": 6.6, "iso": 1}}, {"m1": {"pt": 6.6, "iso": 1}, "m2": {"pt": 7.7, "iso": 2}}, {"m1": {"pt": 6.6, "iso": 1}, "m2": {"pt": 8.8, "iso": 3}}, {"m1": {"pt": 6.6, "iso": 1}, "m2": {"pt": 9.9, "iso": 4}}], [{"m1": {"pt": 7.7, "iso": 2}, "m2": {"pt": 6.6, "iso": 1}}, {"m1": {"pt": 7.7, "iso": 2}, "m2": {"pt": 7.7, "iso": 2}}, {"m1": {"pt": 7.7, "iso": 2}, "m2": {"pt": 8.8, "iso": 3}}, {"m1": {"pt": 7.7, "iso": 2}, "m2": {"pt": 9.9, "iso": 4}}], [{"m1": {"pt": 8.8, "iso": 3}, "m2": {"pt": 6.6, "iso": 1}}, {"m1": {"pt": 8.8, "iso": 3}, "m2": {"pt": 7.7, "iso": 2}}, {"m1": {"pt": 8.8, "iso": 3}, "m2": {"pt": 8.8, "iso": 3}}, {"m1": {"pt": 8.8, "iso": 3}, "m2": {"pt": 9.9, "iso": 4}}], [{"m1": {"pt": 9.9, "iso": 4}, "m2": {"pt": 6.6, "iso": 1}}, {"m1": {"pt": 9.9, "iso": 4}, "m2": {"pt": 7.7, "iso": 2}}, {"m1": {"pt": 9.9, "iso": 4}, "m2": {"pt": 8.8, "iso": 3}}, {"m1": {"pt": 9.9, "iso": 4}, "m2": {"pt": 9.9, "iso": 4}}]]}]

    output, counter = run(r"""
grouped = muons as m1 cross muons as m2 group by m2
""", thedata)
    assert tolist(output) == [{"grouped": [[{"m1": {"pt": 1.1, "iso": 0}, "m2": {"pt": 1.1, "iso": 0}}, {"m1": {"pt": 2.2, "iso": 0}, "m2": {"pt": 1.1, "iso": 0}}, {"m1": {"pt": 3.3, "iso": 100}, "m2": {"pt": 1.1, "iso": 0}}], [{"m1": {"pt": 1.1, "iso": 0}, "m2": {"pt": 2.2, "iso": 0}}, {"m1": {"pt": 2.2, "iso": 0}, "m2": {"pt": 2.2, "iso": 0}}, {"m1": {"pt": 3.3, "iso": 100}, "m2": {"pt": 2.2, "iso": 0}}], [{"m1": {"pt": 1.1, "iso": 0}, "m2": {"pt": 3.3, "iso": 100}}, {"m1": {"pt": 2.2, "iso": 0}, "m2": {"pt": 3.3, "iso": 100}}, {"m1": {"pt": 3.3, "iso": 100}, "m2": {"pt": 3.3, "iso": 100}}]]}, {"grouped": []}, {"grouped": [[{"m1": {"pt": 4.4, "iso": 50}, "m2": {"pt": 4.4, "iso": 50}}, {"m1": {"pt": 5.5, "iso": 30}, "m2": {"pt": 4.4, "iso": 50}}], [{"m1": {"pt": 4.4, "iso": 50}, "m2": {"pt": 5.5, "iso": 30}}, {"m1": {"pt": 5.5, "iso": 30}, "m2": {"pt": 5.5, "iso": 30}}]]}, {"grouped": [[{"m1": {"pt": 6.6, "iso": 1}, "m2": {"pt": 6.6, "iso": 1}}, {"m1": {"pt": 7.7, "iso": 2}, "m2": {"pt": 6.6, "iso": 1}}, {"m1": {"pt": 8.8, "iso": 3}, "m2": {"pt": 6.6, "iso": 1}}, {"m1": {"pt": 9.9, "iso": 4}, "m2": {"pt": 6.6, "iso": 1}}], [{"m1": {"pt": 6.6, "iso": 1}, "m2": {"pt": 7.7, "iso": 2}}, {"m1": {"pt": 7.7, "iso": 2}, "m2": {"pt": 7.7, "iso": 2}}, {"m1": {"pt": 8.8, "iso": 3}, "m2": {"pt": 7.7, "iso": 2}}, {"m1": {"pt": 9.9, "iso": 4}, "m2": {"pt": 7.7, "iso": 2}}], [{"m1": {"pt": 6.6, "iso": 1}, "m2": {"pt": 8.8, "iso": 3}}, {"m1": {"pt": 7.7, "iso": 2}, "m2": {"pt": 8.8, "iso": 3}}, {"m1": {"pt": 8.8, "iso": 3}, "m2": {"pt": 8.8, "iso": 3}}, {"m1": {"pt": 9.9, "iso": 4}, "m2": {"pt": 8.8, "iso": 3}}], [{"m1": {"pt": 6.6, "iso": 1}, "m2": {"pt": 9.9, "iso": 4}}, {"m1": {"pt": 7.7, "iso": 2}, "m2": {"pt": 9.9, "iso": 4}}, {"m1": {"pt": 8.8, "iso": 3}, "m2": {"pt": 9.9, "iso": 4}}, {"m1": {"pt": 9.9, "iso": 4}, "m2": {"pt": 9.9, "iso": 4}}]]}]

@pytest.mark.parametrize("dataset", [test_dataset, test_dataset_awkward])
def test_tabular_max(dataset):
    thedata = dataset()

    output, counter = run(r"""
x = 3
extreme = muons max by pt
""", thedata)
    assert tolist(output) == [{"x": 3, "extreme": {"pt": 3.3, "iso": 100}}, {"x": 3}, {"x": 3, "extreme": {"pt": 5.5, "iso": 30}}, {"x": 3, "extreme": {"pt": 9.9, "iso": 4}}]

    output, counter = run(r"""
x = 3
extreme = muons min by pt
""", thedata)
    assert tolist(output) == [{"x": 3, "extreme": {"pt": 1.1, "iso": 0}}, {"x": 3}, {"x": 3, "extreme": {"pt": 4.4, "iso": 50}}, {"x": 3, "extreme": {"pt": 6.6, "iso": 1}}]

    output, counter = run(r"""
extreme = muons min by pt
whatever = has extreme
""", thedata)
    assert tolist(output) == [{"extreme": {"pt": 1.1, "iso": 0}, "whatever": True}, {"whatever": False}, {"extreme": {"pt": 4.4, "iso": 50}, "whatever": True}, {"extreme": {"pt": 6.6, "iso": 1}, "whatever": True}]

    output, counter = run(r"""
extreme = muons min by pt
whatever = if has extreme then extreme.iso else -100
""", thedata)
    assert tolist(output) == [{"extreme": {"pt": 1.1, "iso": 0}, "whatever": 0}, {"whatever": -100}, {"extreme": {"pt": 4.4, "iso": 50}, "whatever": 50}, {"extreme": {"pt": 6.6, "iso": 1}, "whatever": 1}]

    output, counter = run(r"""
x = 3
extreme = muons min by pt
whatever = if has extreme then extreme.iso
""", thedata)
    assert tolist(output) == [{"x": 3, "extreme": {"pt": 1.1, "iso": 0}, "whatever": 0}, {"x": 3}, {"x": 3, "extreme": {"pt": 4.4, "iso": 50}, "whatever": 50}, {"x": 3, "extreme": {"pt": 6.6, "iso": 1}, "whatever": 1}]

def test_tabular_max_with_identities():
    thedata = test_dataset()

    output, counter = run(r"""
extreme = muons where iso > 2 with { iso2 = 2*iso } union muons min by pt
whatever = if has extreme then extreme?.iso2
""", thedata)
    assert output.tolist() == [{"extreme": {"pt": 1.1, "iso": 0}}, {"extreme": {"pt": 4.4, "iso": 50, "iso2": 100}, "whatever": 100}, {"extreme": {"pt": 6.6, "iso": 1}}]

    output, counter = run(r"""
extreme = muons where iso > 2 with { iso2 = 2*iso } union muons min by pt
whatever = ?extreme?.iso2
""", thedata)
    assert output.tolist() == [{"extreme": {"pt": 1.1, "iso": 0}}, {"extreme": {"pt": 4.4, "iso": 50, "iso2": 100}, "whatever": 100}, {"extreme": {"pt": 6.6, "iso": 1}}]

@pytest.mark.parametrize("dataset", [test_dataset_realistic, test_dataset_realistic_awkward])
def test_realistic_queries(dataset):
    thedata = dataset()
    output, counter = run(r"""
def mass(one, two) {
    91.2 + one.pt + two.pt # yes this is just a stand in
}
best_leptons = {
    eles = electrons as (lep1, lep2) where lep1.charge != lep2.charge
    mus = muons as (lep1, lep2) where lep1.charge != lep2.charge
    Z = eles union mus min by abs(mass(lep1, lep2) - 91.2)
    [?Z.lep1, ?Z.lep2]
}
""", thedata)
    assert tolist(output) == [{'best_leptons': [{'pt': 1, 'charge': 1, 'iso': 10}, {'pt': 2, 'charge': -1, 'iso': 10}]}, {'best_leptons': []}, {'best_leptons': [{'pt': 4.4, 'charge': 1, 'iso': 50}, {'pt': 5.5, 'charge': -1, 'iso': 30}]}, {'best_leptons': [{'pt': 1, 'charge': 1, 'iso': 9}, {'pt': 2, 'charge': -1, 'iso': 8}]}]

@pytest.mark.parametrize("dataset", [test_dataset, test_dataset_awkward])
def test_reducers(dataset):
    thedata = dataset()
    
    output, counter = run(r"""
x = count(muons);
y = count(stuff);
""", thedata)

    assert(tolist(output) == [{'x': 3, 'y': 0}, {'x': 0, 'y': 1}, {'x': 2, 'y': 2}, {'x': 4, 'y': 3}])

    output, counter = run(r"""
xx = sum(muons.pt);
yy = sum(stuff);
""", thedata)

    assert(tolist(output) == [{'xx': 6.6, 'yy': 0}, {'xx': 0, 'yy': 1}, {'xx': 9.9, 'yy': 4}, {'xx': 33.0, 'yy': 9}])

    output, counter = run(r"""
xxx = if(count(muons) > 0) then min(muons.pt) else 0;
yyy = if(count(stuff) > 0) then min(stuff) else 0;
""", thedata)

    assert(tolist(output) == [{'xxx': 1.1, 'yyy': 0}, {'xxx': 0, 'yyy': 1}, {'xxx': 4.4, 'yyy': 2}, {'xxx': 6.6, 'yyy': 3}])

    output, counter = run(r"""
xxxx = if(count(muons) > 0) then max(muons.pt) else 0;
yyyy = if(count(stuff) > 0) then max(stuff) else 0;
""", thedata)

    assert(tolist(output) == [{'xxxx': 3.3, 'yyyy': 0}, {'xxxx': 0, 'yyyy': 1}, {'xxxx': 5.5, 'yyyy': 2}, {'xxxx': 9.9, 'yyyy': 3}])

    output, counter = run(r"""
xxxxx = { temp = muons with { passes = pt > 2 };
          any(temp.passes) }
yyyyy = { temp = muons with { passes = pt > 2 };
          any(temp.passes) }
""", thedata)

    assert(tolist(output) == [{'xxxxx': True, 'yyyyy': True}, {'xxxxx': False, 'yyyyy': False}, {'xxxxx': True, 'yyyyy': True}, {'xxxxx': True, 'yyyyy': True}])

    output, counter = run(r"""
xxxxxx = { temp = muons with { passes = pt > 2 };
           all(temp.passes) }
yyyyyy = { temp = muons with { passes = pt > 2 };
           all(temp.passes) }
""", thedata)
    
    assert(tolist(output) == [{'xxxxxx': False, 'yyyyyy': False}, {'xxxxxx': True, 'yyyyyy': True}, {'xxxxxx': True, 'yyyyyy': True}, {'xxxxxx': True, 'yyyyyy': True}])

@pytest.mark.parametrize("dataset", [test_dataset, test_dataset_awkward])
def test_hist(dataset):
    thedata = dataset()

    output, counter = run(r"""
hist met
""", thedata)
    assert tolist(output) == []
    assert (counter.entries, counter.value, counter.error) == (4, 4.0, 2.0)
    assert counter.keys() == ["0"]
    counts, edges = counter["0"].numpy()
    assert counts.tolist() == [1, 0, 0, 1, 0, 0, 1, 0, 0, 1]
    assert edges.tolist() == [100, 130, 160, 190, 220, 250, 280, 310, 340, 370, 400]

    output, counter = run(r"""
hist met by regular(10, 0, 1000)
""", thedata)
    counts, edges = counter["0"].numpy()
    assert counts.tolist() == [0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    assert edges.tolist() == [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    output, counter = run(r"""
hist met by regular(10, 0, 1000), met - 1 by regular(5, 0, 500)
""", thedata)
    counts, (xedges, yedges) = counter["0"].numpy()
    assert counts.tolist() == [[0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]
    assert xedges.tolist() == [0.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0]
    assert yedges.tolist() == [0.0, 100.0, 200.0, 300.0, 400.0, 500.0]

    output, counter = run(r"""
hist met by regular(10, 0, 1000) weight by 2
""", thedata)
    counts, edges = counter["0"].numpy()
    assert counts.tolist() == [0, 2, 2, 2, 2, 0, 0, 0, 0, 0]
    assert edges.tolist() == [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    output, counter = run(r"""
hist met by regular(10, 0, 1000) named "one" titled "two"
""", thedata)
    assert counter["one"].title == "two"

@pytest.mark.parametrize("dataset", [test_dataset, test_dataset_awkward])
def test_hist_dot(dataset):
    thedata = dataset()
    
    output, counter = run(r"""
hist { muons max by pt }.pt
""", thedata)
    assert tolist(output) == []
    assert (counter.entries, counter.value, counter.error) == (4, 4.0, 2.0)
    assert counter.keys() == ["0"]
    counts, edges = counter["0"].numpy()
    assert counts.tolist() == [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    assert edges.tolist() == [3.3, 3.96, 4.62, 5.279999999999999, 5.9399999999999995, 6.6, 7.26, 7.92, 8.58, 9.24, 9.9]
    
    output, counter = run(r"""
leading_mu = muons max by pt
hist ?leading_mu.pt
""", thedata)
    assert tolist(output) == [{'leading_mu': {'pt': 3.3, 'iso': 100}}, {'leading_mu': {'pt': 5.5, 'iso': 30}}, {'leading_mu': {'pt': 9.9, 'iso': 4}}]
    assert (counter.entries, counter.value, counter.error) == (4, 4.0, 2.0)
    assert counter.keys() == ["0"]
    counts, edges = counter["0"].numpy()
    assert counts.tolist() == [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    assert edges.tolist() == [3.3, 3.96, 4.62, 5.279999999999999, 5.9399999999999995, 6.6, 7.26, 7.92, 8.58, 9.24, 9.9]

@pytest.mark.parametrize("dataset", [test_dataset, test_dataset_awkward])
def test_cutvary(dataset):
    thedata = dataset()

    output, counter = run(r"""
vary by { x = 1 } by { x = 2 } by { x = 3 } {
    hist x by regular(5, 0, 5)
}
""", thedata)
    assert counter.allkeys() == ["0", "0/0", "1", "1/0", "2", "2/0"]
    assert counter["0/0"].numpy()[0].tolist() == [0, 4, 0, 0, 0]
    assert counter["1/0"].numpy()[0].tolist() == [0, 0, 4, 0, 0]
    assert counter["2/0"].numpy()[0].tolist() == [0, 0, 0, 4, 0]

    output, counter = run(r"""
vary by { x = 1 } named "one" by { x = 2 } named "two" by { x = 3 } named "three" {
    hist x by regular(5, 0, 5)
}
""", thedata)
    assert counter.allkeys() == ["one", "one/0", "two", "two/0", "three", "three/0"]
    assert counter["one/0"].numpy()[0].tolist() == [0, 4, 0, 0, 0]
    assert counter["two/0"].numpy()[0].tolist() == [0, 0, 4, 0, 0]
    assert counter["three/0"].numpy()[0].tolist() == [0, 0, 0, 4, 0]

    output, counter = run(r"""
cut met > 200 {
    hist met by regular(5, 0, 500)
}
""", thedata)
    assert counter.allkeys() == ["0", "0/0"]
    assert counter["0/0"].numpy()[0].tolist() == [0, 0, 0, 1, 1]

    output, counter = run(r"""
cut met > 200 weight by 2 {
    hist met by regular(5, 0, 500)
}
""", thedata)
    assert counter.allkeys() == ["0", "0/0"]
    assert counter["0/0"].numpy()[0].tolist() == [0, 0, 0, 2, 2]

    output, counter = run(r"""
cut met > 200 named "one" titled "two" {
    hist met by regular(5, 0, 500)
}
""", thedata)
    assert counter.allkeys() == ["one", "one/0"]
    assert counter["one/0"].numpy()[0].tolist() == [0, 0, 0, 1, 1]
    assert counter["one"].title == "two"

