"""Regression test for issue #4: StreamGraph.most_connected_n must run.

The objective ``self.connectedness`` is already a bound method, so passing an
extra ``args=(self,)`` to ``differential_evolution`` produced an arg-count
mismatch that crashed the optimizer on modern scipy. With that removed the
optimizer runs to completion. The optimizer is stochastic, so this asserts on
shape / no-raise rather than exact values.
"""

from grapht.graph import StreamGraph


def _small_stream_graph(max_dim=6):
    graph = StreamGraph(max_dim)
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 2), (1, 3)]:
        graph.append(a, b)
    return graph


def test_most_connected_n_runs_and_returns_expected_shape():
    graph = _small_stream_graph()

    n = 3
    result = graph.most_connected_n(n)

    assert len(result.x) == n
