import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
pca = PCA(n_components=2)
pca.fit(X)

print(pca.explained_variance_ratio_)

pca = PCA(n_components=2, svd_solver='full')
pca.fit(X)

print(pca.explained_variance_ratio_)

pca = PCA(n_components=1, svd_solver='arpack')
pca.fit(X)

print(pca.explained_variance_ratio_)

X = np.array([2.5,.5,2.2,1.9,3.1,2.3,2,1,1.5,1.1])
y = np.array([2.4,0.7,2.9,2.2,3.0,2.7,1.6,1.1,1.6,.9])

plt.scatter(X,y)
#plt.show()

X_mean = X - X.mean()
y_mean = y - y.mean()

plt.figure()
plt.scatter(X_mean, y_mean)
#plt.show()

matrix = np.vstack((X_mean,y_mean))
print(matrix)

print(np.cov(matrix))

pca = PCA(n_components=2)

pca.fit(np.cov(matrix))

print (pca.explained_variance_ratio_)

print(pca.components_)


plt.figure()
plt.scatter(X_mean,y_mean)

t = np.arange(-2.,2,0.2)
plt.plot(t, pca.components_[0,0] + pca.components_[0,1]*t,'r')
plt.plot(t, pca.components_[1,0] + pca.components_[1,1]*t,'g')
#plt.show()

svd = TruncatedSVD()

svd.fit(matrix)

print(svd.components_)
