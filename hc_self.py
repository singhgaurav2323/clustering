# hierarchial clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [2,3]].values

#using dentrogram to find optimal number of cluster
import scipy.cluster.hierarchy as spy 
dedogram=spy.dendrogram(spy.linkage(X,method='ward'))
plt.title('dendogram')
plt.xlabel('customer')
plt.ylabel('Euclidean distance')
plt.show()

#fitting hierechical clustering to mall data
from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=5,affinity="euclidean",linkage='ward')
yhc=hc.fit_predict(X)

#visualising cluster
plt.scatter(X[yhc == 0,0],X[yhc ==0,1],s=100,c='red',label='carefull')
plt.scatter(X[yhc == 1,0],X[yhc ==1,1],s=100,c='brown',label='standard')
plt.scatter(X[yhc == 2,0],X[yhc ==2,1],s=100,c='blue',label='target')
plt.scatter(X[yhc == 3,0],X[yhc ==3,1],s=100,c='green',label='careless')
plt.scatter(X[yhc == 4,0],X[yhc ==4,1],s=100,c='yellow',label='sensible')
plt.title("customer data")
plt.xlabel('annual income k$')
plt.ylabel('rank label(1-100)')
plt.legend()
plt.show()