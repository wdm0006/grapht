"""StreamGraph.append rejects out-of-range indices without touching the matrix."""

import numpy as np
import pytest

from grapht import StreamGraph


@pytest.mark.parametrize("a, b", [(-1, 0), (0, -1), (-4, 0), (0, -4), (-1, -1)])
def test_negative_index_raises_and_leaves_matrix_unchanged(a, b):
    g = StreamGraph(4)
    g.append(1, 2)
    before = g.get_dense().copy()
    with pytest.raises(IndexError):
        g.append(a, b)
    assert np.array_equal(g.get_dense(), before)
    expected = np.zeros((4, 4), dtype=np.int8)
    expected[1, 2] = 1
    assert np.array_equal(g.get_dense(), expected)


def test_last_valid_index_still_works():
    g = StreamGraph(4)
    g.append(3, 0)
    expected = np.zeros((4, 4), dtype=np.int8)
    expected[3, 0] = 1
    assert np.array_equal(g.get_dense(), expected)


@pytest.mark.parametrize("a, b", [(4, 0), (0, 4)])
def test_index_at_max_dim_still_raises(a, b):
    g = StreamGraph(4)
    with pytest.raises(IndexError):
        g.append(a, b)
    assert not g.get_dense().any()
