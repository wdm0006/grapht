"""Tests for selecting valid node indices with StreamGraph.most_connected_n."""

import numpy as np
import pytest

from grapht.graph import StreamGraph


def _small_stream_graph(max_dim=6):
    graph = StreamGraph(max_dim)
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 2), (1, 3)]:
        graph.append(a, b)
    return graph


@pytest.mark.parametrize("n", [1, 3, 6])
def test_most_connected_n_returns_valid_distinct_node_indices(n):
    graph = _small_stream_graph()

    result = graph.most_connected_n(n)

    assert len(result.x) == n
    assert np.all(result.x == np.floor(result.x))
    assert np.all((0 <= result.x) & (result.x < graph.max_dim))
    assert len(set(result.x)) == n


@pytest.mark.parametrize("n", [0, -1, 7])
def test_most_connected_n_rejects_invalid_count_before_optimization(monkeypatch, n):
    graph = _small_stream_graph()
    optimizer_called = False

    def fail_if_called(*args, **kwargs):
        nonlocal optimizer_called
        optimizer_called = True

    monkeypatch.setattr("grapht.graph.differential_evolution", fail_if_called)

    with pytest.raises(ValueError, match="n must be between 1 and max_dim"):
        graph.most_connected_n(n)

    assert not optimizer_called
