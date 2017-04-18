#!/bin/usr/env python2.7

import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from dateutil.rrule import rrule, DAILY
from operator import itemgetter
from sklearn.decomposition import PCA

# AU Adjacency Matrix
AU_DG = nx.DiGraph()
AU_DG.add_nodes_from(range(0, 10))
AU_DG.add_weighted_edges_from([(2,3,5),(5,6,5),(2,4,9),(3,6,8),(1,3,5)])
# AU_adjacency_matrix = nx.adjacency_matrix(AU_DG)
AU_adjacency_matrix = nx.to_numpy_matrix(AU_DG)
vectorSize = AU_adjacency_matrix.shape[0] * AU_adjacency_matrix.shape[1]
AU_matrix = np.zeros((3,vectorSize))
AU_adjacency_vector = AU_adjacency_matrix.reshape(1,vectorSize)
AU_matrix[0,:] = AU_adjacency_vector
AU_matrix[1,:] = AU_adjacency_vector
AU_matrix[2,:] = AU_adjacency_vector

# AU Adjacency Matrix
LT_DG = nx.DiGraph()
LT_DG.add_nodes_from(range(0, 10))
LT_DG.add_weighted_edges_from([(1,2,5),(3,2,5),(6,4,9),(7,9,8),(0,2,5)])
LT_adjacency_matrix = nx.to_numpy_matrix(LT_DG)
vectorSize = LT_adjacency_matrix.shape[0] * LT_adjacency_matrix.shape[1]
LT_matrix = np.zeros((3,vectorSize))
LT_adjacency_vector = LT_adjacency_matrix.reshape(1,vectorSize)
LT_matrix[0,:] = LT_adjacency_vector
LT_matrix[1,:] = LT_adjacency_vector
LT_matrix[2,:] = LT_adjacency_vector

AU_cols_sum = np.sum(AU_matrix, axis=0)
LT_cols_sum = np.sum(LT_matrix, axis=0)

mask = (AU_cols_sum != 0) | (LT_cols_sum != 0)
AU_matrix = AU_matrix.compress(mask, axis=1)
LT_matrix = LT_matrix.compress(mask, axis=1)

print (AU_matrix)

print (LT_matrix)