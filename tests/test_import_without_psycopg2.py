"""Regression tests for issue #1: grapht must import without psycopg2 installed.

psycopg2 is only needed by ``StreamGraph.from_psql``; importing the library or
building the core graph types must not require it.
"""

import builtins
import sys

import numpy as np


def _block_psycopg2(monkeypatch):
    """Make any ``import psycopg2`` raise ModuleNotFoundError."""
    for name in list(sys.modules):
        if name == 'psycopg2' or name.startswith('psycopg2.'):
            monkeypatch.delitem(sys.modules, name, raising=False)

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == 'psycopg2' or name.startswith('psycopg2.'):
            raise ModuleNotFoundError("No module named 'psycopg2'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', fake_import)


def test_import_and_core_graphs_work_without_psycopg2(monkeypatch):
    _block_psycopg2(monkeypatch)

    # Force a fresh import of the package with psycopg2 unavailable.
    for name in list(sys.modules):
        if name == 'grapht' or name.startswith('grapht.'):
            monkeypatch.delitem(sys.modules, name, raising=False)

    import grapht
    from grapht.graph import DictGraph, DenseGraph, StreamGraph

    dict_graph = DictGraph({0: [1], 1: [0, 2], 2: [1]})
    assert dict_graph.get_dense().shape == (3, 3)

    dense_graph = DenseGraph(np.eye(3, dtype=np.int8))
    assert dense_graph.get_dense().shape == (3, 3)

    stream_graph = StreamGraph(3)
    stream_graph.append(0, 1)
    assert stream_graph.get_dense()[0, 1] == 1

    assert grapht.DictGraph is DictGraph


def test_from_psql_raises_import_error_when_psycopg2_missing(monkeypatch):
    _block_psycopg2(monkeypatch)

    from grapht.graph import StreamGraph

    graph = StreamGraph(3)
    try:
        graph.from_psql('user', 'pw', 'db', 'localhost', 'schema', 'table')
    except ModuleNotFoundError as exc:
        assert 'psycopg2' in str(exc)
    else:
        raise AssertionError('from_psql should require psycopg2 when it is missing')
