"""connectedness() is the negative size of the union of the subset's out-neighbourhoods."""

import numpy as np

from grapht.graph import DenseGraph


def _graph():
    # 0 -> {1, 2}, 1 -> {2}, 2 -> {}, 3 -> {0}
    return DenseGraph(
        np.array(
            [
                [0, 1, 1, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 0],
                [1, 0, 0, 0],
            ],
            dtype=np.int8,
        )
    )


def test_single_node():
    assert _graph().connectedness([0]) == -2.0


def test_overlapping_neighbourhoods_do_not_double_count():
    # {1, 2} union {2} == {1, 2}
    assert _graph().connectedness([0, 1]) == -2.0


def test_disjoint_neighbourhoods_add():
    # {1, 2} union {0} == {0, 1, 2}
    assert _graph().connectedness([0, 3]) == -3.0


def test_node_without_connections():
    assert _graph().connectedness([2]) == 0.0
