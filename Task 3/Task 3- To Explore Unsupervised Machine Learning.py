#!/usr/bin/env python
# coding: utf-8

# # From the given ‘Iris’ dataset, predict the optimum number of clusters and represent it visually.

# ## Importing all the necessary libraries

# In[80]:


import numpy as np
import pandas as pd
import plotly.express as px
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Read Data

# In[81]:


dataset=pd.read_csv("C:/Users/Omkar/Downloads/Iris.csv")
dataset


# In[82]:


dataset.describe()


# In[83]:


dataset.info()


# In[84]:


# Checking Names Of species
dataset['Species'].unique()


# In[85]:


x1=dataset.iloc[:,:-1].values
y1=dataset.iloc[:,-1].values

ly=LabelEncoder()
y=ly.fit_transform(y)


# ### Visualizing Data

# In[86]:


fig_scatter=px.scatter_matrix(dataset,dimensions=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'],color="Species")
fig_scatter.update_layout(title='Iris Scatter Plots')
fig_scatter.show()


# ### Creating Model

# #### Model implemented using KMeans Clustering Algorithm from skLearn Package

# In[87]:


# Setting X and Y
x = dataset [ [ 'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm' ] ].values
y = dataset['Species'].values


# In[88]:


# Removing redundant data
dataset.drop('Species',axis=1,inplace=True)
dataset.drop('Id',axis=1,inplace=True)


# In[89]:


dataset


# In[90]:


wcss = []            # Within cluster sum of squares
for i in range(1, 15):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', 
                    max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    
# Plotting the results onto a line graph to get a number of clusters

plt.plot(range(1, 15), wcss)
plt.title('The Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')


# In[91]:


# 3 appears as "the elbow"
# Applying kmeans to the dataset
kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x)


# #### Creating Model for k=3

# In[92]:


# Visualising the clusters

plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], 
            s = 100, c = 'violet', label = 'Iris-setosa')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], 
            s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1],
            s = 100, c = 'green', label = 'Iris-virginica')

# Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], 
            s = 100, c = 'red', label = 'Centroids')

plt.legend()


# In[96]:


data=pd.read_csv("C:/Users/Omkar/Downloads/Iris.csv")


# In[97]:


fig=px.scatter_3d(data,x='SepalLengthCm',y='SepalWidthCm',z='PetalWidthCm',color='Species',symbol='Species',size='PetalLengthCm',title="3-D view")
fig.show()

