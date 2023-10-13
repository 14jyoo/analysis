#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# In[2]:


df=pd.read_csv(r"C:\Users\JYOTHIKA\Downloads\titanic.csv")


# In[3]:


df.head(10)


# In[4]:


df.info()


# In[5]:


df.shape


# In[6]:


df.describe()


# In[7]:


# Survival distribution by gender
sns.countplot(x='Survived', hue='Sex', data=df)
plt.show()


# In[8]:


# Handle missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Pclass'].fillna(df['Pclass'].mode()[0], inplace=True)

# Encode categorical variables
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['Pclass'] = le.fit_transform(df['Pclass'])


print(df.head())


# In[10]:


# Feature Engineering Example

# Create a new feature 'FamilySize' by combining 'Siblings/Spouses Aboard' and 'Parents/Children Aboard'
df['FamilySize'] = df['Siblings/Spouses Aboard'] + df['Parents/Children Aboard'] + 1

# Create a new feature 'IsAlone' to indicate whether a passenger is traveling alone
df['IsAlone'] = 0
df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1

# Create a new feature 'Title' by extracting titles from 'Name'
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

# Group rare titles into 'Other'
df['Title'] = df['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')

# Map common titles to numerical values
title_mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Other': 5}
df['Title'] = df['Title'].map(title_mapping)

# Drop unnecessary columns
df = df.drop(['Name', 'Siblings/Spouses Aboard', 'Parents/Children Aboard'], axis=1)

# Print the modified dataset with new features
print(df.head())


# In[11]:


X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[12]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


print("Training set - Features:", X_train.shape)
print("Training set - Target:", y_train.shape)
print("Testing set - Features:", X_test.shape)
print("Testing set - Target:", y_test.shape)


# In[15]:


import seaborn as sns
import matplotlib.pyplot as plt

# A histogram of the 'Age' feature before splitting
sns.histplot(X['Age'], kde=True)
plt.title('Age Distribution Before Splitting')
plt.show()


# In[16]:


import seaborn as sns
import matplotlib.pyplot as plt

# A histogram of the 'Age' feature in the training set
sns.histplot(X_train['Age'], kde=True)
plt.title('Age Distribution in Training Set')
plt.show()


# In[ ]:




