#!/usr/bin/env python
# coding: utf-8

# # For the given ‘Iris’ dataset, create the Decision Tree classifier and visualize it graphically. The purpose is if we feed any new data to this classifier, it would be able to predict the right class accordingly.

# ### Importing all the necessary libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Read Data

# In[2]:


df=pd.read_csv('D:\Programs\Data sets\Iris.csv')


# In[3]:


#Gets first 5 rows
df.head()


# In[4]:


# Basic Info about csv
df.info()


# In[5]:


# Basic Statistics on csv
df.describe()


# In[6]:


df.shape


# In[7]:


df.notnull().sum()


# In[8]:


# To check value counts of Species column
df["Species"].value_counts()


# In[9]:


# Pair Plot
sns.pairplot(df, hue="Species", palette="viridis", diag_kind='kde')


# In[10]:


# Converting categorical variables into numeric
df["Species"] = df["Species"].astype('category')
df["Species"] = df["Species"].cat.codes
df["Species"].value_counts()


# In[11]:


# Classifying independent and dependent variables
x = df.iloc[:,1:5].values
y = df.iloc[:,-1].values

print("x=\n",x,"\n\n","y=\n",y)


# In[12]:


# Splitting the data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x , y , test_size=0.2 , random_state=0)


# In[13]:


# Training the model
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier().fit(x_train , y_train)


# In[14]:


# Prediction
pred = dtree.predict(x_test)


# In[15]:


# Confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,pred)


# In[16]:


features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
classes = ['setosa','versicolor','virginica']


# In[17]:


# Vizualising the decision tree
fig = plt.figure(figsize=(25,16))
Viztree = tree.plot_tree(dtree, feature_names=features, class_names=classes, filled=True)

