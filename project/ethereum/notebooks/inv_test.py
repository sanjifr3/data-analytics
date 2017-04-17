import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from dateutil.rrule import rrule, DAILY
from operator import itemgetter
from sklearn.decomposition import PCA

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
AU_matrix = np.zeros((vectorSize, num_days))
LT_matrix = np.zeros((vectorSize, num_days))
matrix_idx = 0

for day in rrule(DAILY, dtstart=first_date, until=last_date):
    date = day.strftime("%Y-%m-%d")
    print('progress:',matrix_idx/num_days*100,'%')
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
    AU_adjacency_vector = AU_adjacency_matrix.reshape(vectorSize,1)
    print (AU_adjacency_vector)
    AU_matrix[:,matrix_idx] = AU_adjacency_vector[0]


    # LT Adjacency Matrix
    LT_DG = nx.DiGraph()
    LT_DG.add_nodes_from(range(0,maxID+1))
    LT_DG.add_weighted_edges_from(LT_daily_tuples)
    # #LT_adjacency_matrix = nx.adjacency_matrix(LT_DG)
    LT_adjacency_matrix = nx.to_numpy_matrix(LT_DG)
    LT_adjacency_vector = LT_adjacency_matrix.reshape(vectorSize,1)
    LT_matrix[:,matrix_idx]  = LT_adjacency_vector[0]
    

    if False:
    #if day.day == 8:
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

print (LT_matrix)
print (AU_matrix)
#
# # Get indices and columns from LT_matrix (same for both matrices)
# index = range(0,LT_matrix.shape[0])
# columns = range(0, LT_matrix.shape[1])
#
# # Convert numpy matrix to Pandas dataframe
# LT_pdDF = pd.DataFrame(data=LT_matrix, index=index, columns=columns)
# AU_pdDF = pd.DataFrame(data=AU_matrix, index=index, columns=columns)
#
# LT_pdDF.to_csv('LT_pca_matrix.csv',index=False)
# AU_pdDF.to_csv('AU_pca_matrix.csv',index=False)
#
#
# #plt.show()
#
# pca = PCA(n_components=2,svd_solver='full')
# pca.fit(LT_matrix)
# print('\n')
# print(pca.explained_variance_)
# print(pca.explained_variance_ratio_)
# print(pca.components_)

