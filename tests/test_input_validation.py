import numpy as np
import pytest

from grapht import DenseGraph, DictGraph


@pytest.mark.parametrize('arr', [np.ones((2, 3)), np.ones(4), np.ones((2, 2, 2))])
def test_dense_rejects_non_square(arr):
    with pytest.raises(ValueError) as exc:
        DenseGraph(arr)
    assert str(arr.shape) in str(exc.value)


def test_dense_valid_unchanged():
    m = np.array([[0, 1], [0, 0]])
    assert (DenseGraph(m).get_dense() == m).all()


def test_dict_rejects_out_of_range_connection():
    with pytest.raises(ValueError) as exc:
        DictGraph({0: [5]})
    assert '5' in str(exc.value)


def test_dict_rejects_negative_connection():
    with pytest.raises(ValueError) as exc:
        DictGraph({0: [-1], 1: []})
    assert '-1' in str(exc.value)


def test_dict_rejects_bad_keys():
    with pytest.raises(ValueError) as exc:
        DictGraph({-1: [0], 0: []})
    assert '-1' in str(exc.value)
    with pytest.raises(ValueError) as exc:
        DictGraph({'a': []})
    assert "'a'" in str(exc.value)


def test_dict_valid_unchanged():
    g = DictGraph({0: [1], 1: [0, 2], 2: [1]})
    assert g.get_dense().tolist() == [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
