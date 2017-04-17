import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

num_users = 10  #1639
num_transactions = 20 #4837957
num_unique_transactions = 3  #4333

data = 100+20*np.random.randn(num_transactions,4)
data[:,:2] = np.random.randint(1,num_users,(num_transactions,2))
data[:,2] = np.random.randint(1,num_unique_transactions+1,(num_transactions))

data_df = pd.DataFrame(data, columns=['From','To','TransactionID','Value'])

data_df['New From'] = [''] * num_transactions

id_itr = 100
for Tid in data_df['TransactionID'].unique():
    if (data_df[(data_df['TransactionID'] == Tid) & (data_df['New From'] != '')].empty):
        data_df.ix[data_df['TransactionID'] == Tid, 'New From'] = id_itr
        id_itr += 100

    else:
        temp_id = id_itr
        temp_id = data_df[(data_df['TransactionID'] == Tid) & (data_df['New From'] != '')]['New From'].values[0]
        data_df.ix[data_df['TransactionID'] == Tid, 'New From'] = temp_id

    print ('\n','\n',Tid)
    print(data_df)




#rows = 300
#columns = 100
#X = np.random.randn(rows, columns)
#
# ## Normalize so each row sums to 1
# for i in range(X.shape[0]):
#     X[i,:] = X[i,:] / X[i,:].sum()
#
# ## Preprocess so average is removed from each column
# for i in range(X.shape[1]):
#     X[:,i] = X[:,i] - X[:,i].mean()
#
# ## SVD
# u,s,v = np.linalg.svd(X,full_matrices=True)
#
# S = np.zeros((rows,columns))
# S[:s.shape[0], :s.shape[0]] = np.diag(s)
# #print(np.allclose(X, np.dot(u, np.dot(S,v))))
# #print(np.dot(u, np.dot(S,v)))
#
# ## PCA
# pca = PCA(n_components=2, svd_solver='full')
# pca.fit(X)
# #print(pca.explained_variance_)
# print(pca.explained_variance_ratio_)
# print(pca.explained_variance_ratio_.sum()*100)
# print(pca.components_.shape)
#

#print(pca.components_.T)
#print(v[:,0:2])


# pca = PCA(n_components=1,svd_solver='full')
# pca.fit(X)
# print('\n')
# print(pca.explained_variance_)
# print(pca.explained_variance_ratio_)
# #print(pca.components_)

# pca = PCA(n_components=2,svd_solver='full')
# pca.fit(X)
# print('\n')
# print(pca.explained_variance_)
# print(pca.explained_variance_ratio_)
# #print(pca.components_)

# pca = PCA(n_components=3,svd_solver='full')
# pca.fit(X)
# print('\n')
# print(pca.explained_variance_)
# print(pca.explained_variance_ratio_)
# print(pca.components_)

# pca = PCA(n_components=4,svd_solver='full')
# pca.fit(X)
# print('\n')
# print(pca.explained_variance_)
# print(pca.explained_variance_ratio_)
# #print(pca.components_)