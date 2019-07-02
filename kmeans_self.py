# K-mean algorithum


#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing database
dataset=pd.read_csv('Mall_Customers.csv')
X= dataset.iloc[:,[2,3]].values

#using elbow method to find optimal number of cluster
from sklearn.cluster import KMeans
cmss=[]
for i in range (1,11):
    kmeans=KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300)
    kmeans.fit(X)
    cmss.append(kmeans.inertia_)

plt.plot(range(1,11),cmss)
plt.title('elbow curve')
plt.xlabel('number of cluster')
plt.ylabel('cmss value')
plt.show()

#applying k mean to mall data
kmeans=KMeans(n_clusters=4, init='k-means++', n_init=10, max_iter=300)
ykmeans=kmeans.fit_predict(X)

#plotting the clustterd data
plt.scatter(X[ykmeans == 0,0],X[ykmeans == 0,1],s=100,c='red',label='cluster 1')
plt.scatter(X[ykmeans == 1,0],X[ykmeans == 1,1],s=100,c='yellow',label='cluster 2')
plt.scatter(X[ykmeans == 2,0],X[ykmeans == 2,1],s=100,c='blue',label='cluster 3')
plt.scatter(X[ykmeans == 3,0],X[ykmeans == 3,1],s=100,c='green',label='cluster 4')

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='black',label='centroid')
plt.xlabel('annual income K$')
plt.ylabel('ranking(1=100)')
plt.legend()
plt.show()
