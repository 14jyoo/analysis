#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


data =pd.read_csv(r"C:\Users\JYOTHIKA\Downloads\archive (13)\creditcard.csv")


# In[3]:


data.head()


# In[5]:


dir(data)


# In[5]:


data.head()


# In[6]:


data.tail()


# In[7]:


data.shape


# In[8]:


print("Number of columns: {}".format(data.shape[1]))
print("Number of rows: {}".format(data.shape[0]))


# In[9]:


data.info()


# In[13]:


data.groupby(['Class']).count()


# In[14]:


x= data.drop(['Time','Class'],axis=1).values
y = data.iloc[:,-1]


# In[15]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)


# In[16]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[17]:


model.fit(x_train,y_train)


# In[18]:


model.score(x_test,y_test)


# In[19]:


model.score(x_train,y_train)


# In[20]:


from sklearn.metrics import classification_report
y_pred = model.predict(x_test)


# In[21]:


print(classification_report(y_test,y_pred))


# In[22]:


get_ipython().system('pip install imblearn')


# In[23]:


from imblearn.over_sampling import SMOTE


# In[24]:


upsample=SMOTE()


# In[25]:


x,y = upsample.fit_resample(x,y)


# In[26]:


from collections import Counter


# In[27]:


count = Counter(y)


# In[28]:


print(count)


# In[29]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)


# In[30]:


model.fit(x_train,y_train)


# In[31]:


model.score(x_train,y_train)


# In[32]:


model.score(x_test,y_test)


# In[33]:


y_pred = model.predict(x_test)


# In[34]:


print(classification_report(y_pred,y_test))


# In[35]:


get_ipython().system('pip freeze > requirements.txt')


# In[36]:


import pickle

# Save the trained model to a file
with open("Credit.pkl", "wb") as file:
    pickle.dump(model, file)


# In[ ]:




