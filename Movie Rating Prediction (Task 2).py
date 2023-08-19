#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


#loading dataset
movies = pd.read_csv("movies.csv", delimiter='::', engine='python')
movies


# In[3]:


#checking information of the given dataset
movies.info()


# In[4]:


#checking dimension of dataset
movies.shape


# In[5]:


#checking for the null values in the dataset
movies.isnull().sum()


# In[6]:


#viewing the last 10 rows of the dataset
movies.tail(10)


# In[7]:


#loading the dataset
ratings = pd.read_csv("ratings.csv", delimiter='::', engine='python')
ratings


# In[8]:


#checking information of the dataset
ratings.info()


# In[9]:


#checking the dimension of the dataset
ratings.shape


# In[10]:


#checking for the null values in the dataset
ratings.isnull().sum()


# In[11]:


#viewing the last 10 rows of the dataset
ratings.tail(10)


# In[12]:


#loading dataset
users = pd.read_csv("users.csv", delimiter='::', engine='python')
users


# In[13]:


#converting string values to integer
users['Gender'] = users['Gender'].map({'M':0,'F':1})
users.head()


# In[14]:


#checking the info, dimension and null values in the dataset
users.info()
users.shape
users.isnull().sum()


# In[15]:


#viewing last 10 rows of the dataset
users.tail(10)


# In[18]:


#distinct values of ID 
unique_counts = ratings['ID'].nunique()
print("Number of unique values in'{}': {}".format('ID',unique_counts))


# In[19]:


# min and max values of ID in Ratings
min_ratings = ratings['ID'].min()
print("Min rating value  in '{}':{}".format('ID',min_ratings))

max_ratings = ratings['ID'].max()
print("Max rating value in '{}':{}".format('ID',max_ratings))


# In[20]:


#distinct values of ID 
unique_counts_ID = ratings['ID'].nunique()
print("Number of unique values in '{}': {}".format('ID',unique_counts_ID))


# In[25]:


df = pd.merge(movies, ratings, on=["ID", "ID"])
print(df.head())


# In[26]:


# merge movies,ratings and users table
data = pd.concat([df,users], axis=1)
data.head()


# In[27]:


#checking info after merging
data.info()


# In[28]:


# checking dimension of the dataset
data.shape


# In[29]:


# checking for any null values
data.isnull().sum()


# In[40]:


#grouping data as ID and Ratings
ratings_counts = data.groupby(['ID','Ratings']).size().reset_index(name = 'UserCount')


# In[37]:


# setting threshold value which is used to get those movies which have rating more than 100
threshold = 100
new_df = ratings_counts[ratings_counts['UserCount']>=threshold]
new_df


# In[41]:


new_data = pd.merge(new_df,data[['ID','Ratings','Gender','Age','Occupation']])
new_data.head(15)


# In[42]:


new_data.shape


# In[59]:


#importing libraries 
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC


# In[44]:


#Distribution of ratings of all movies
ratings = data["Ratings"].value_counts()
numbers = ratings.index
quantity = ratings.values
import plotly.express as px
fig = px.pie(data, values=quantity, names=numbers)
fig.show()


# In[45]:


#Visualize the overall rating by users
new_data['Ratings'].value_counts().plot(kind='barh',alpha=0.7,figsize=(10,10))
plt.show()


# In[46]:


#splitting of data into training and testing set
X = new_data.drop(["Gender","Age","Occupation"],axis=1)
y = new_data['Ratings']


# In[47]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3, random_state = 45)


# In[48]:


#model selection
model = LinearRegression()


# In[49]:


#model fitting
model.fit(X_train,y_train)


# In[50]:


y_pred = model.predict(X_test)
mse = mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mse)
print("Mean sqaured error:",mse)
print("Root mean sqaured error:",rmse)


# In[52]:


model2 = DecisionTreeClassifier()


# In[53]:


model2.fit(X_train,y_train)


# In[58]:


Y_pred = model2.predict(X_test)
acc = accuracy_score(y_test,Y_pred)*100
print("Accuracy Score:",acc)


# In[ ]:




