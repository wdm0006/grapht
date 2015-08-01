__author__ = 'willmcginnis'
import random
import time

from grapht.graph import DictGraph, StreamGraph

if __name__ == '__main__':
    print('DictGraph')
    g = {0: [2], 1: [2, 3], 2: [1, 3], 3: [1, 2, 4, 6], 4: [3, 5], 5: [4, 6, 7], 6: [3, 4, 5], 7: [5, 8], 8: [7], 9: [8], 10: [9], 11: [8, 9], 12: [11], 13: [12], 14: [13], 15: [1]}
    gp = DictGraph(g)

    print('Original Adjacency Matrix')
    print(gp.get_dense())

    print('Second Connections')
    print(gp.get_n_connection(n=2).toarray())

    print('Third Connections')
    print(gp.get_n_connection(n=3).toarray())

    print('\n\nStream Graph')
    gp2 = StreamGraph(max_dim=10000001)
    for _ in range(1000000):
        a = random.randint(0, 10000000)
        b = random.randint(0, 10000000)
        gp2.append(a, b)

    print('Number of non-zero elements')
    print(gp2.get_nnz())

    print('Calculating 4th Degree connections for a 1,000,000 edge, 10,000,000 node graph')
    start_time = time.time()
    temp = gp2.get_n_connection(n=4)
    elapsed = time.time() - start_time
    print('TIME: %s' % (str(elapsed), ))
