"""Tests for the core, pure graph operations (issue #3).

These exercise construction (``DictGraph``, ``DenseGraph``, ``StreamGraph``) and
the read/query methods (``get_n_connection``, ``get_connections``, ``get_dense``)
on tiny hand-checkable graphs. No network or Postgres is involved.
"""

import numpy as np

from grapht.graph import DenseGraph, DictGraph, StreamGraph


def test_dict_graph_builds_symmetric_adjacency():
    # Path 0-1-2-3. DictGraph sets both [k, con] and [con, k], so the matrix
    # is symmetric even though only one direction is listed per key.
    graph = DictGraph({0: [1], 1: [0, 2], 2: [1, 3], 3: [2]})

    expected = np.array(
        [
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
        ],
        dtype=np.int8,
    )
    np.testing.assert_array_equal(graph.get_dense(), expected)


def test_dense_graph_round_trips_matrix():
    # A directed 3-node chain 0->1->2; DenseGraph should preserve it verbatim.
    adj = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=np.int8)
    graph = DenseGraph(adj)

    np.testing.assert_array_equal(graph.get_dense(), adj)


def test_stream_graph_append_sets_directed_entries():
    graph = StreamGraph(3)
    graph.append(0, 1)
    graph.append(1, 2)

    expected = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=np.int8)
    np.testing.assert_array_equal(graph.get_dense(), expected)


def test_get_connections_returns_node_row():
    graph = DictGraph({0: [1], 1: [0, 2], 2: [1, 3], 3: [2]})

    # Node 1 connects to 0 and 2.
    np.testing.assert_array_equal(
        graph.get_connections(1).toarray(), np.array([[1, 0, 1, 0]], dtype=np.int8)
    )


def test_get_n_connection_one_hop_on_path():
    graph = DictGraph({0: [1], 1: [0, 2], 2: [1, 3], 3: [2]})

    # One extra hop reaches the neighbour-of-a-neighbour; diagonal stays 0.
    expected = np.array(
        [
            [0, 1, 1, 0],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [0, 1, 1, 0],
        ],
        dtype=np.int8,
    )
    result = graph.get_n_connection(n=1).toarray()
    np.testing.assert_array_equal(result, expected)
    assert np.all(np.diag(result) == 0)


def test_get_n_connection_two_hops_on_path():
    graph = DictGraph({0: [1], 1: [0, 2], 2: [1, 3], 3: [2]})

    expected = np.array(
        [
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
        ],
        dtype=np.int8,
    )
    result = graph.get_n_connection(n=2).toarray()
    np.testing.assert_array_equal(result, expected)
    assert np.all(np.diag(result) == 0)
