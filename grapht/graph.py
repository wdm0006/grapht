__author__ = 'willmcginnis'
from scipy.sparse import lil_matrix, triu
from scipy.optimize import differential_evolution
import numpy as np
import copy
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import random

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
        return delta.sign()

    def get_dense(self):
        """

        """

        return self.a.toarray()

    def get_connections(self, k):
        """

        """

        return self.a.getrow(k)

    def get_nnz(self):
        """

        """

        return self.a.getnnz()

    @staticmethod
    def connectedness(subset, self):
        """
        Returns the relative connectedness of a subset of nodes.

        """
        connections = self.get_connections(int(subset[0])) * -1.0
        for idx, node in enumerate(subset):
            if idx > 0:
                connections = connections.minimum(self.get_connections(int(node)) * -1.0)
        return connections.sum()

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
        self.max_dim = max_dim
        self.a = lil_matrix((max_dim, max_dim), dtype=np.int8)

    def append(self, a, b):
        """
        Add and edge between nodes a and b to the network.

        :param a: node index a (int)
        :param b: node index b (int)
        :return:

        """

        self.a[a, b] = 1

    def most_connected_n(self, n=10):
        bounds = [(0, self.max_dim) for _ in range(n)]
        result = differential_evolution(self.connectedness, bounds=bounds, args=(self, ), maxiter=10, popsize=25)
        return result

    def from_psql(self, username, password, database, host, schema, table, follower='follower', followee='followee'):
        """
        Will create a graph from a postgresql table with 2 columns.

        :param username:
        :param password:
        :param database:
        :param host:
        :param schema:
        :param table:
        :return:
        """

        conn = psycopg2.connect(database=database, user=username, password=password, host=host, connect_timeout=60)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        curr = conn.cursor()

        sql = 'SELECT %s, %s from %s.%s' % (follower, followee, schema, table)
        curr.execute(sql)
        for a, b in curr.fetchall():
            self.append(a, b)

        curr.close()
        conn.close()

        return self

