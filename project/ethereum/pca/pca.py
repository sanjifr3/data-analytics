import numpy as np
from sklearn.decomposition import PCA

rows = 20
columns = 50
X = np.random.randn(rows, columns)

## Normalize so each row sums to 1
for i in range(X.shape[0]):
    X[i,:] = X[i,:] / X[i,:].sum()

## Preprocess so average is removed from each column
for i in range(X.shape[1]):
    X[:,i] = X[:,i] - X[:,i].mean()

## SVD
u,s,v = np.linalg.svd(X,full_matrices=True)

S = np.zeros((rows,columns))
S[:s.shape[0], :s.shape[0]] = np.diag(s)
#print(np.allclose(X, np.dot(u, np.dot(S,v))))
#print(np.dot(u, np.dot(S,v)))

## PCA
pca = PCA(n_components=2, svd_solver='full')
pca.fit(X)
#print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
print(pca.components_.shape)
print(v[:,0].shape)

print(pca.components_.T)
print(v[:,0:2])


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