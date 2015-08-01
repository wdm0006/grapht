__author__ = 'willmcginnis'
from scipy.sparse import lil_matrix, triu
import numpy as np
import copy

class BaseGraph(object):

    a = None

    def get_n_connection(self, n=2):
        """

        """

        delta = copy.deepcopy(self.a)

        for _ in range(n):
            delta = delta.dot(self.a).sign()

        delta = delta + self.a
        delta.setdiag(0, k=0)
        return triu(delta).sign()

    def get_dense(self):
        """

        """

        return triu(self.a).toarray()

    def get_connections(self, k):
        """

        """

        return self.a.getrow(k)

    def get_nnz(self):
        """

        """

        return self.a.getnnz()

    def __repr__(self):
        return str(self.a)

    def __str__(self):
        return str(self.a)


class DictGraph(BaseGraph):
    """
    """
    def __init__(self, graph_dict):
        """
        An object for creating graphs from a dictionary of the form: {node: [connections]}.

        :param graph_dict: dict
        :return:

        """

        self.a = self.from_dict(graph_dict)

    @staticmethod
    def from_dict(graph):
        """
        Assembles the graph from a dictionary of type {node: [connections]}

        :param graph: dict
        :return: sparse matrix

        """

        a = lil_matrix((len(graph.keys()), len(graph.keys())), dtype=np.int8)

        for key in graph.keys():
            for con in graph[key]:
                a[key, con] = 1
                a[con, key] = 1

        return a

class StreamGraph(BaseGraph):
    def __init__(self, max_dim):
        """
        An object for streaming graphs, allows you to use .append notation to add edges in dynamically. All nodes must be
        named for integers in the range (0, max_dim).

        :param max_dim: the maximum dimension of the network to be streamed in
        :return:

        """
        self.a = lil_matrix((max_dim, max_dim), dtype=np.int8)

    def append(self, a, b):
        """
        Add and edge between nodes a and b to the network.

        :param a: node index a (int)
        :param b: node index b (int)
        :return:

        """

        self.a[a, b] = 1
        self.a[b, a] = 1

