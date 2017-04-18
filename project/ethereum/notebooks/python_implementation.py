#!/bin/usr/env python2.7
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from dateutil.rrule import rrule, DAILY
from operator import itemgetter
from sklearn.decomposition import PCA
# from pyspark.mllib.common import callMLlibFunc, JavaModelWrapper
# from pyspark.mllib.linalg.distributed import RowMatrix
# from pyspark.ml.feature import *
# from pyspark.mllib.linalg import Vectors

# class SVD(JavaModelWrapper):
#     """
#     Wrapper aruond the SVD Scala case class
#     """

#     @property
#     def U(self):
#         """
#         Returns a RowMatrix whose columns are the left singular values of the 
#         SVD if computeU was set to True.
#         """
#         u = self.call("U")
#         if u is not None:
#             return RowMatrix(u)

#     @property
#     def s(self):
#         """
#         Returns a DenseVector with singular values in descending order.
#         """
#         return self.call("s")

#     @property
#     def V(self):
#         """ 
#         Returns a DenseMatrix whose columns are the right singular vectors 
#         of the SVD.
#         """
#         return self.call("V")

# def computeSVD(row_matrix, k, computeU=False, rCond=1e-9):
#     """
#     Computes the singular value decomposition of the RowMatrix.
#     The given row matrix A of dimension (m X n) is decomposed into U * s * V'T where
#     * s: DenseVector consisting of square root of the eigenvalues (singular values) in descending order.
#     * U: (m X k) (left singular vectors) is a RowMatrix whose columns are the eigenvectors of (A X A')
#     * v: (n X k) (right singular vectors) is a Matrix whose columns are the eigenvectors of (A' X A)
#     :param k: number of singular values to keep. We might return less than k if there are numerically zero singular values.
#     :param computeU: Whether of not to compute U. If set to be True, then U is computed by A * V * sigma^-1
#     :param rCond: the reciprocal condition number. All singular values smaller than rCond * sigma(0) are treated as zero, where sigma(0) is the largest singular value.
#     :returns: SVD object
#     """
#     java_model = row_matrix._java_matrix_wrapper.call("computeSVD", int(k), computeU, float(rCond))
#     return SVD(java_model)


# data = [(Vectors.dense([0.0, 1.0, 0.0, 7.0, 0.0]),), (Vectors.dense([2.0, 0.0, 3.0, 4.0, 5.0]),), (Vectors.dense([4.0, 0.0, 0.0, 6.0, 7.0]),)]
# df = sqlContext.createDataFrame(data,["features"])

# pca_extracted = PCA(k=2, inputCol="features", outputCol="pca_features")

# model = pca_extracted.fit(df)
# features = model.transform(df) # this create a DataFrame with the regular features and pca_features

# # We can now extract the pca_features to prepare our RowMatrix.
# pca_features = features.select("pca_features").rdd.map(lambda row : row[0])
# mat = RowMatrix(pca_features)

# # Once the RowMatrix is ready we can compute our Singular Value Decomposition
# svd = computeSVD(mat,2,True)
# svd.s
# # DenseVector([9.491, 4.6253])
# svd.U.rows.collect()
# # [DenseVector([0.1129, -0.909]), DenseVector([0.463, 0.4055]), DenseVector([0.8792, -0.0968])]
# svd.V
# # DenseMatrix(2, 2, [-0.8025, -0.5967, -0.5967, 0.8025], 0)


# exit(0)
draw = False
skip = True

if skip == False:
    LT_core = pd.read_csv('./data/lt_core_matrix.csv',sep=',')
    AU_core = pd.read_csv('./data/au_core_matrix.csv',sep=',')

    maxID = max(LT_core['fromID'].max(),
                LT_core['toID'].max(),
                AU_core['fromID'].max(),
                AU_core['toID'].max())

    first_date = min(LT_core['date'].min(), AU_core['date'].min())
    last_date = max(LT_core['date'].max(), AU_core['date'].max())
    first_date =  date(int(first_date[:4]), int(first_date[5:7]), int(first_date[8:11]))
    last_date =  date(int(last_date[:4]), int(last_date[5:7]), int(last_date[8:11]))
    #last_date = date(2015,8,9)

    num_days = (last_date-first_date).days+1
    vectorSize = (maxID+1)*(maxID+1)
    AU_matrix = np.zeros((num_days, vectorSize))
    LT_matrix = np.zeros((num_days, vectorSize))
    matrix_idx = 0

    for day in rrule(DAILY, dtstart=first_date, until=last_date):
        date = day.strftime("%Y-%m-%d")
        print('progress:',matrix_idx/float(num_days)*100,'%')
        AU_daily_tuples = [0] * len(AU_core[AU_core['date'] == date])
        LT_daily_tuples = [0] * len(LT_core[LT_core['date'] == date])
        AU_idx = 0
        LT_idx = 0

        for edge in AU_core[AU_core['date'] == date].itertuples(index=False):
            AU_daily_tuples[AU_idx] = edge[1:4]
            AU_idx += 1

        for edge in LT_core[LT_core['date'] == date].itertuples(index=False):
            LT_daily_tuples[LT_idx] = edge[1:4]
            LT_idx += 1

        # print (date, AU_daily_tuples)
        # print (date, LT_daily_tuples)

        # AU Adjacency Matrix
        AU_DG = nx.DiGraph()
        AU_DG.add_nodes_from(range(0, maxID + 1))
        AU_DG.add_weighted_edges_from(AU_daily_tuples)
        # AU_adjacency_matrix = nx.adjacency_matrix(AU_DG)
        AU_adjacency_matrix = nx.to_numpy_matrix(AU_DG)
        AU_adjacency_vector = AU_adjacency_matrix.reshape(1,vectorSize)
        AU_matrix[matrix_idx,:] = AU_adjacency_vector

        # LT Adjacency Matrix
        LT_DG = nx.DiGraph()
        LT_DG.add_nodes_from(range(0,maxID+1))
        LT_DG.add_weighted_edges_from(LT_daily_tuples)
        # #LT_adjacency_matrix = nx.adjacency_matrix(LT_DG)
        LT_adjacency_matrix = nx.to_numpy_matrix(LT_DG)
        LT_adjacency_vector = LT_adjacency_matrix.reshape(1,vectorSize)
        LT_matrix[matrix_idx,:] = LT_adjacency_vector

        #if False:
        if draw == True and day.day == 8 and (day.month == 8 or day.month == 2):
            # Draw LT Core Network
            LT_edges = []
            maxWeight = max(AU_daily_tuples,key=itemgetter(1))[2]
            for tpl in AU_daily_tuples:
                weightPer = tpl[2]/maxWeight * 100
                if weightPer > 50:
                    LT_edges.append((tpl[0],tpl[1],weightPer))
            LT_size = [x[2] for x in LT_edges]
            LT_nodes = list(set([x[0] for x in LT_edges] + [x[1] for x in LT_edges]))

            G = nx.DiGraph()
            G.add_nodes_from(LT_nodes)
            G.add_weighted_edges_from(LT_edges)

            plt.figure()
            plt.title(str(date) + ' - LT Core Weighted Graph')
            pos = nx.spring_layout(G)
            edge_labels = dict([((u, v,), d['weight'] * maxWeight / 100) for u, v, d in G.edges(data=True)])
            nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'), node_size=LT_size)
            nx.draw_networkx_edges(G, pos, arrows=True)
            nx.draw_networkx_labels(G, pos)
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

            # Draw AU Core Network
            AU_edges = []
            maxWeight = max(AU_daily_tuples,key=itemgetter(1))[2]
            for tpl in AU_daily_tuples:
                weightPer = tpl[2]/maxWeight * 100
                if weightPer > 50:
                    AU_edges.append((tpl[0],tpl[1],weightPer))
            AU_size = [x[2] for x in AU_edges]
            AU_nodes = list(set([x[0] for x in AU_edges] + [x[1] for x in AU_edges]))

            G = nx.DiGraph()
            G.add_nodes_from(AU_nodes)
            G.add_weighted_edges_from(AU_edges)

            plt.figure()
            plt.title(str(date) + ' - AU Core Weighted Graph')
            pos = nx.spring_layout(G)
            edge_labels = dict([((u, v,), d['weight'] * maxWeight / 100) for u, v, d in G.edges(data=True)])
            nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'), node_size=LT_size)
            nx.draw_networkx_edges(G, pos, arrows=True)
            nx.draw_networkx_labels(G, pos)
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

        matrix_idx += 1

    AU_cols_sum = np.sum(AU_matrix, axis=0)
    LT_cols_sum = np.sum(LT_matrix, axis=0)

    mask = (AU_cols_sum != 0) | (LT_cols_sum != 0)
    AU_matrix_compressed = AU_matrix.compress(mask, axis=1)
    LT_matrix_compressed = LT_matrix.compress(mask, axis=1)

    # np.save("./data/LT_pca_comp_matrix.csv",LT_matrix_compressed)
    # np.save("./data/AU_pca_comp_matrix.csv",AU_matrix_compressed)
    # np.save("./data/LT_pca_comp_matrix",LT_matrix_compressed)
    # np.save("./data/AU_pca_comp_matrix",AU_matrix_compressed)

AU_matrix_comp = np.load("./data/AU_pca_comp_matrix.npy")
LT_matrix_comp = np.load("./data/LT_pca_comp_matrix.npy")

## Normalize so each row sums to 1
for i in range(AU_matrix_comp.shape[0]):
    AU_matrix_comp[i,:] = AU_matrix_comp[i,:] / AU_matrix_comp[i,:].sum()

## Preprocess so average is removed from each column
for i in range(AU_matrix_comp.shape[1]):
    AU_matrix_comp[:,i] = AU_matrix_comp[:,i] - AU_matrix_comp[:,i].mean()

## Normalize so each row sums to 1
for i in range(LT_matrix_comp.shape[0]):
    LT_matrix_comp[i,:] = LT_matrix_comp[i,:] / LT_matrix_comp[i,:].sum()

## Preprocess so average is removed from each column
for i in range(LT_matrix_comp.shape[1]):
    LT_matrix_comp[:,i] = LT_matrix_comp[:,i] - LT_matrix_comp[:,i].mean()

# np.save("./data/LT_pca_comp_norm_matrix",LT_matrix_comp)
# np.save("./data/AU_pca_comp_norm_matrix",AU_matrix_comp)

print (LT_matrix_comp)
print (AU_matrix_comp)

# Get indices and columns from LT_matrix (same for both matrices)
index = range(0,LT_matrix_comp.shape[0])
columns = range(0, LT_matrix_comp.shape[1])

# Convert numpy matrix to Pandas dataframe
LT_pdDF = pd.DataFrame(data=LT_matrix_comp, index=index, columns=columns)
AU_pdDF = pd.DataFrame(data=AU_matrix_comp, index=index, columns=columns)

from sklearn.decomposition import PCA

pca = PCA(n_components = 386, whiten=False)
pca.fit(AU_matrix_comp)

print (pca.components_.transpose())
print (pca.transform(AU_matrix_comp))
print (pca.explained_variance_)
print (pca.explained_variance_ratio_)

from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=2)

svd.fit(AU_matrix_comp)

print(svd.explained_variance_ratio_)
print (svd.fit_transform(AU_matrix_comp))


# LT_pdDF.to_csv('./data/LT_pca_comp_norm_matrix.csv',index=False, header=False)
# AU_pdDF.to_csv('./data/AU_pca_comp_norm_matrix.csv',index=False, header=False)


# plt.show()
#
# pca = PCA(n_components=2,svd_solver='full')
# pca.fit(LT_matrix)
# print('\n')
# print(pca.explained_variance_)
# print(pca.explained_variance_ratio_)
# print(pca.components_)

