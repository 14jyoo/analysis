#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install tensorflow


# In[2]:


# Import libraries
import numpy as np
import pandas as pd
from datetime import datetime 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


# In[3]:


df=pd.read_csv(r"C:\Users\JYOTHIKA\Downloads\archive (5)\WMT.csv")


# In[4]:


#gives you all the statistical information from the dataset
df.describe()


# In[5]:


df.info()


# In[6]:


df.shape


# In[7]:


df.head(10)


# In[8]:


df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date')

# Extracting the column "Close" to predict
column_name = 'Close'
data = df[column_name].values.reshape(-1, 1)


# In[9]:


print("Basic information about the dataset:-")
df.info()


# In[10]:


print("Shape of the processed data:")
data.shape


# In[11]:


# Normalize the data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Displays the first few rows of the normalized data
print("\nFirst few rows of the normalized data:")
print(data_scaled[:10])


# In[12]:


# Creating sequences for LSTM
sequence_length = 10
X, y = [], []

for i in range(len(data_scaled) - sequence_length):
    X.append(data_scaled[i:i + sequence_length, 0])
    y.append(data_scaled[i + sequence_length, 0])

X, y = np.array(X), np.array(y)

# Train-test split
split_ratio = 0.8
split = int(split_ratio * len(X))
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

# Reshape for LSTM input shape
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))


# In[13]:


# Display the shapes of the created sequences
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)


# In[14]:


import matplotlib.pyplot as plt

# Plot the original time series data
plt.figure(figsize=(10, 5))
plt.plot(data_scaled, label='Original Data')
plt.title('Walmart Stock Price: Original Data')
plt.xlabel('Time')
plt.ylabel('Normalized Close Price')
plt.legend()
plt.show()


# In[15]:


# Plot a few sequences from the training set
plt.figure(figsize=(10, 5))
for i in range(5):  # Plotting 5 sequences
    plt.plot(X_train[i], label=f'Sequence {i + 1}')

plt.title('Walmart Stock Price: Sequences for Training')
plt.xlabel('Time Steps')
plt.ylabel('Normalized Close Price')
plt.legend()
plt.show()
#each sequence is a subset of our original time series data.


# In[16]:


# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Print the model summary
print(model.summary())


# In[17]:


# Train the model
history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

# Plot training loss and validation loss over epochs
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Training Progress')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[18]:


# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform predictions to original scale
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
y_train_original = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))


# In[19]:


import matplotlib.pyplot as plt

# Plot results
plt.figure(figsize=(15, 6))

# Training Set: Actual vs Predicted
plt.subplot(1, 2, 1)
plt.plot(df.index[:len(y_train_original)], y_train_original, label='Actual')
plt.plot(df.index[:len(y_train_original)], train_predict, label='Predicted')
plt.title('Training Set: Actual vs Predicted')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()


# In[20]:


# Plot results
plt.figure(figsize=(15, 6))

# Test Set: Actual vs Predicted
plt.subplot(1, 2, 2)
test_index_start = len(y_train_original) + sequence_length  # Adjusted the start index for the test set
plt.plot(df.index[test_index_start:], y_test_original, label='Actual')
plt.plot(df.index[test_index_start:], test_predict, label='Predicted')
plt.title('Test Set: Actual vs Predicted')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()

plt.tight_layout()  # Adjust layout for better presentation
plt.show()


# In[21]:


from sklearn.metrics import mean_squared_error

# Evaluate the model
train_rmse = np.sqrt(mean_squared_error(y_train_original, train_predict))
test_rmse = np.sqrt(mean_squared_error(y_test_original, test_predict))

print(f'Training RMSE: {train_rmse}')
print(f'Test RMSE: {test_rmse}')


# In[22]:


# Plot training loss and validation loss over epochs
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Training Progress')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[ ]:




