from scipy.sparse import lil_matrix
from scipy.optimize import differential_evolution
import numpy as np
import copy

__author__ = 'willmcginnis'


class BaseGraph(object):
    """

    Base object for a graph, contains most interfaces with a graph once it's been created. Various other classes will
    inherit from this in order to gain that, while retaining differentiation in how the graph is created. The core
    functionality here is the storing of a simple graph in a scipy sparse adjacency matrix.

    """

    a = None

    def get_n_connection(self, n=2):
        """
        This function will calculate the n-hop version of the adjacency matrix using a simple chained dot-product. The
        diagonal will always be 0s.

        Every path length from 1 through ``n + 1`` is accumulated, so an entry is non-zero whenever one node can reach
        another in at most ``n + 1`` hops. ``n=0`` returns the direct adjacency; each further increment of ``n`` adds one
        more reachable hop without dropping the shorter ones.

        Returns a scipy sparse matrix.

        """

        power = copy.deepcopy(self.a)
        delta = copy.deepcopy(self.a)

        for _ in range(n):
            power = power.dot(self.a).sign()
            delta = (delta + power).sign()

        delta.setdiag(0, k=0)
        return delta.sign()

    def get_dense(self):
        """
        Returns the adjacency matrix of the submitted graph in dense form (be careful with high dimension graphs).

        """

        return self.a.toarray()

    def get_connections(self, k):
        """
        Returns a sparse vector indicting all connections that a certain node has.

        """

        return self.a.getrow(k)

    def get_nnz(self):
        """
        Returns the number of edges in the graph (number of off-diagonal
        nonzero entries in the adjacency matrix). This is a pure read and does
        not modify the stored graph.

        """

        return self.a.getnnz() - int(np.count_nonzero(self.a.diagonal()))

    def connectedness(self, subset):
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


class DenseGraph(BaseGraph):
    """
    """
    def __init__(self, adj_matrix):
        """
        An object for creating graphs from a dictionary of the form: {node: [connections]}.

        :param graph_dict: dict
        :return:

        """

        self.a = lil_matrix(adj_matrix)


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
        if n < 1 or n > self.max_dim:
            raise ValueError("n must be between 1 and max_dim")

        def distinct_connectedness(subset):
            if len(set(subset)) != n:
                return np.inf
            return self.connectedness(subset)

        bounds = [(0, self.max_dim - 1) for _ in range(n)]
        result = differential_evolution(
            distinct_connectedness,
            bounds=bounds,
            integrality=True,
            x0=np.arange(n),
            maxiter=10,
            popsize=25,
        )
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

        import psycopg2
        from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

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
