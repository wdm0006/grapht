"""Regression tests for issue #2: get_nnz() must not mutate the graph.

Counting edges is a read; it must not zero the diagonal (which would destroy
self-loops) or otherwise alter the stored adjacency matrix.
"""

import numpy as np

from grapht.graph import DictGraph, DenseGraph


def test_get_nnz_does_not_mutate_graph_with_self_loop():
    # Node 0 has a self-loop (diagonal entry).
    graph = DictGraph({0: [0, 1], 1: [0]})
    before = graph.get_dense()

    graph.get_nnz()

    after = graph.get_dense()
    np.testing.assert_array_equal(before, after)


def test_get_nnz_is_stable_across_repeated_calls():
    graph = DictGraph({0: [0, 1], 1: [0]})
    first = graph.get_nnz()
    second = graph.get_nnz()
    third = graph.get_nnz()

    assert first == second == third


def test_get_nnz_counts_off_diagonal_edges():
    # Symmetric edge 0-1 stores two nonzeros; the self-loop on the diagonal
    # must not be counted.
    graph = DictGraph({0: [0, 1], 1: [0]})
    assert graph.get_nnz() == 2


def test_get_nnz_matches_total_for_diagonal_free_graph():
    adj = np.array([[0, 1, 0],
                    [1, 0, 1],
                    [0, 1, 0]])
    graph = DenseGraph(adj)
    assert graph.get_nnz() == int(np.count_nonzero(adj))
