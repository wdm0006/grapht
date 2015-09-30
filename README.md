Grapht
======

A toy project for exploring graphs in python with scipy sparse matrices.  Not intended to be real science or for any kind
of actual production use.

Features
--------

Graphs can be created using a few different interfaces:

 * DenseGraph: takes in a dense numpy adjacency matrix
 * StreamGraph: uses .append() notation to allow you to iterative build a graph by adding edges. Includes from_psql function to build from a database table.
 * DictGraph: assembles a graph from a dictionary of nodes and connections.
 
Once assembled, you can do a few things (not a ton):

 * get the n-hop equivalent adjacency matrix of the graph
 * get a dense adjacency matrix
 * get the connections for any node as a sparse vector
 * get the total number of edges
 * get the relative connectedness of a collection of nodes

Example
-------

More detail at: `this site <http://willmcginnis.com/2015/09/29/grapht-graph-connectedness-and-dimensionality/>`_

Code:

    from grapht.graph import DenseGraph
    import random
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.style.use('ggplot')
    
    output = []
    con_per_con = 5
    
    for dim in range(1, 2000):
        # Make an array
        density = float(dim * con_per_con) / float(dim * dim)
        A = np.array([[1 if random.random() < density else 0 for _ in range(dim)] for _ in range(dim)])
    
        # Instantiate the graph
        gp = DenseGraph(A)
    
        # Calc the as-is ratio
        before_ratio = gp.get_nnz() / float(dim * dim)
    
        # Calc the the 1-hop adj
        hop_matrix = gp.get_n_connection(n=2).toarray()
        gp2 = DenseGraph(hop_matrix)
        after_ratio = gp2.get_nnz() / float(dim * dim)
    
        # Calc the the 2-hop adj
        two_hop_matrix = gp.get_n_connection(n=3).toarray()
        gp3 = DenseGraph(two_hop_matrix)
        two_ratio = gp3.get_nnz() / float(dim * dim)
        output.append([before_ratio, after_ratio, two_ratio])
        print(dim)
    
    df = pd.DataFrame(output, columns=['Zero Hop', 'One Hop', 'Two Hop'])
    
    ax = df.plot()
    plt.title('Connectedness of Adjacency Matrices Based On Dimension\n(%s connections per node held constant)' % con_per_con)
    plt.xlabel('Dimension (number of nodes)')
    plt.ylabel('Connectedness (percentage of non-zero entries)')
    plt.show()

Plot: 

![grapht](https://github.com/wdm0006/grapht/blob/master/plots/connectedness.png "Grapht Example Plot")


License
-------

BSD: see license.md