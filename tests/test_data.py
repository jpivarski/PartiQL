import pytest

def test_data():
    from awkwardql.data import (RecordArray,
                                PrimitiveArray,
                                ListArray,
                                UnionArray,
                                instantiate)
    
    # data in columnar form
    events = RecordArray({
        "muons": ListArray([0, 3, 3, 5], [3, 3, 5, 9], RecordArray({
            "pt": PrimitiveArray([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]),
            "iso": PrimitiveArray([0, 0, 100, 50, 30, 1, 2, 3, 4])
        })),
        "jets": ListArray([0, 5, 6, 8], [5, 6, 8, 12], RecordArray({
            "pt": PrimitiveArray([1, 2, 3, 4, 5, 100, 30, 50, 1, 2, 3, 4]),
            "mass": PrimitiveArray([10, 10, 10, 10, 10, 5, 15, 15, 9, 8, 7, 6])
        })),
        "met": PrimitiveArray([100, 200, 300, 400])
    })

    # same data in rowwise form
    assert events == [
        {'muons': [
            {'pt': 1.1, 'iso': 0},
            {'pt': 2.2, 'iso': 0},
            {'pt': 3.3, 'iso': 100}],
         'jets': [
            {'pt': 1, 'mass': 10},
            {'pt': 2, 'mass': 10},
            {'pt': 3, 'mass': 10},
            {'pt': 4, 'mass': 10},
            {'pt': 5, 'mass': 10}],
         'met': 100},
        {'muons': [],
         'jets': [{'pt': 100, 'mass': 5}],
         'met': 200},
        {'muons': [
            {'pt': 4.4, 'iso': 50},
            {'pt': 5.5, 'iso': 30}],
         'jets': [
            {'pt': 30, 'mass': 15},
            {'pt': 50, 'mass': 15}],
         'met': 300},
        {'muons': [
            {'pt': 6.6, 'iso': 1},
            {'pt': 7.7, 'iso': 2},
            {'pt': 8.8, 'iso': 3},
            {'pt': 9.9, 'iso': 4}],
         'jets': [
            {'pt': 1, 'mass': 9},
            {'pt': 2, 'mass': 8},
            {'pt': 3, 'mass': 7},
            {'pt': 4, 'mass': 6}],
         'met': 400}]

    # projection down to the numerical values
    assert events["muons"]["pt"] == [[1.1, 2.2, 3.3], [], [4.4, 5.5], [6.6, 7.7, 8.8, 9.9]]

    # single record object
    assert events[0] == {
        'muons': [
            {'pt': 1.1, 'iso': 0},
            {'pt': 2.2, 'iso': 0},
            {'pt': 3.3, 'iso': 100}],
        'jets': [
            {'pt': 1, 'mass': 10},
            {'pt': 2, 'mass': 10},
            {'pt': 3, 'mass': 10},
            {'pt': 4, 'mass': 10},
            {'pt': 5, 'mass': 10}],
        'met': 100}

    # integer and string indexes commute, but string-string and integer-integer do not
    assert events["muons"][0]          == events[0]["muons"]
    assert events["muons"][0]["pt"]    == events[0]["muons"]["pt"]
    assert events["muons"][0][2]       == events[0]["muons"][2]
    assert events["muons"][0]["pt"][2] == events[0]["muons"]["pt"][2]
    assert events["muons"][0]["pt"][2] == events[0]["muons"][2]["pt"]
    assert events["muons"][0]["pt"][2] == events["muons"][0][2]["pt"]
    assert events["muons"][0]["pt"][2] == events["muons"]["pt"][0][2]

    events.setindex()

    muonpt = events.contents["muons"].content.contents["pt"]
    assert muonpt.row == [(0, 0), (0, 1), (0, 2), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2), (3, 3)]
    assert muonpt.col == ("muons", "pt")

    muoniso = events.contents["muons"].content.contents["iso"]
    assert muonpt.row == muoniso.row
    assert muonpt.row.same(muoniso.row)

    c1, c2 = muonpt.col.tolist()
    for i, (r1, r2) in enumerate(muonpt.row):
        assert events[c1][c2][r1][r2] == muonpt[i]

    instantiate(events)

    egamma = UnionArray([0, 0, 1, 0, 1, 1, 1, 0, 0], [0, 1, 0, 2, 1, 2, 3, 3, 4], [
        RecordArray({
            "q": PrimitiveArray([1, -1, -1, 1, 1]),
            "pt": PrimitiveArray([10, 20, 30, 40, 50])
        }),
        RecordArray({
            "pt": PrimitiveArray([1.1, 2.2, 3.3, 4.4])
        })
    ])

    assert egamma == [
        {'pt': 10, 'q': 1},
        {'pt': 20, 'q': -1},
        {'pt': 1.1},
        {'pt': 30, 'q': -1},
        {'pt': 2.2},
        {'pt': 3.3},
        {'pt': 4.4},
        {'pt': 40, 'q': 1},
        {'pt': 50, 'q': 1}]

    assert egamma["pt"] == [10, 20, 1.1, 30, 2.2, 3.3, 4.4, 40, 50]

    egamma.setindex()
    assert egamma.contents[0].contents["pt"].row == [(0,), (1,), (3,), (7,), (8,)]
    assert egamma.contents[1].contents["pt"].row == [(2,), (4,), (5,), (6,)]
    assert egamma.contents[0].contents["pt"].col == ("pt",)
    assert egamma.contents[1].contents["pt"].col == ("pt",)

    instantiate(egamma)
