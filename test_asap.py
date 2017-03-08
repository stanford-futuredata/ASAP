
import pytest

@pytest.fixture
def js_taxi(fname='demo-taxi-output.csv'):
    import csv
    with open(fname, 'r') as ifh:
        icsv = csv.reader(ifh)
        _ = icsv.next() # head
        return [ float(x[1]) for x in icsv ]

@pytest.fixture
def python_taxi(fname='test.csv'):
    return js_taxi(fname=fname)


def test_zip_diff(js_taxi, python_taxi):
    for j,p in zip(js_taxi,python_taxi):
        assert abs(j - p) < 0.0001

def test_len(js_taxi, python_taxi):
    assert len(js_taxi) == len(python_taxi)
