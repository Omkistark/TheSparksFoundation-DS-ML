#!/usr/bin/env python
# coding: utf-8

# # What will be predicted score if a student study for 9.25 hrs in a day?

# ##  Importing all the necessary libraries

# In[5]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# ###  Read Data

# In[6]:


data=pd.read_csv("https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv")


# In[7]:


data


# ### Clean Data 

# In[8]:


# Cleaning data is not necessary in this case but to generalize this code, it is better to clean the data
data=data.dropna()


# ## Determining x axis and y axis

# In[9]:


x=np.array(data["Hours"]).reshape(-1,1)
y=np.array(data["Scores"]).reshape(-1,1)


# ##  Visualising Data

# In[10]:


plt.scatter(x,y)
plt.title("Input Data")
plt.xlabel("Hours studied")
plt.ylabel("Percentage scored")


# ## Model the Data

# In[11]:


# Generating the best fit for all values
model=LinearRegression().fit(x,y)


# ## Evaluate Model

# In[12]:


# Generating test data and train data, although the train data is not used to model
# Instead the entire data will be used to model and just the test data will be used for evaluation 
x_t,x_test,y_t,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[13]:


model.score(x_test,y_test)*100


# In[14]:


print("Actual values  Predicted values")
y2=model.predict(x_test)
for i in range(0,y2.size):
    print("\t",y_test[i],"\t",y2[i])


# ## Visualization

# In[15]:


plt.scatter(x,y)                                       #Original Data
plt.scatter(x,model.predict(x),color="green")          #Predicted Data
plt.title("Linear Model Hours vs Percentage")
plt.xlabel("Hours studied")
plt.ylabel("Percentage scored")


# ## Final Solution

# In[16]:


model.predict([[9.25]])


# In[17]:


plt.scatter(x,y)                                       #Original Data
plt.plot(x,model.predict(x),color="yellow")                           #Predicted Data
plt.scatter([[9.25]],model.predict([[9.25]]),marker ="s",color="red")
plt.title("Hours vs Percentage")
plt.xlabel("Hours studied")
plt.ylabel("Percentage scored")

