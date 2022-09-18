#!/usr/bin/env python
# coding: utf-8

# In[15]:


#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn import metrics
from sklearn.linear_model import LinearRegression


# In[16]:


pip install openpyxl


# In[79]:


#loading the data from excel
df=pd.read_csv("F:/project1/data.csv")
df


# In[18]:


#information od data
df.info()


# In[19]:


# head is used to select top 5 rows
df.head()


# In[20]:


# tail is used to select bottom 5 rows
df.tail()


# In[21]:


df.isnull().sum()


# In[22]:


# it returns Boolen values
df.isnull()


# In[78]:


sns.set(rc={"figure.figsize":(10,10)})
sns.pairplot(df)


# In[24]:


df.isnull().sum().sum()


# In[25]:


#visualising the data using heatmap
plt.figure(figsize=(10,10))
sns.heatmap(df.corr(),annot=True, cmap='coolwarm')


# In[26]:


# it shows data the in histogram form
sns.distplot(df["Exp"])


# In[27]:


sns.distplot(df["Actual CTC"])


# In[28]:


#Transform  data into categorical variables to the numerical variables
df['Role'].replace(['Manager','Executive'],[0,1],inplace=True)
df


# In[29]:


df1=pd.read_csv("F:/project1/testdata2.csv")
df1


# In[30]:


df1.columns


# In[32]:


df1.info()


# In[33]:


df1.isnull().sum()


# In[31]:


df1.isnull()


# In[35]:


df1.drop(df1.columns[13:24],axis=1,inplace=True)
df1


# In[41]:


df1.info()


# In[36]:


x=df1[["College_T1","College_T2","Role_Manager","City_Metro","previous CTC","previous job changes","Graduation marks"]]
y=df1["Actual CTC"]


# In[37]:


x_train,x_test,y_train,t_test=train_test_split(x,y,train_size=0.4,random_state=100)


# In[38]:


lr=LinearRegression()
lr.fit(x_train,y_train)


# In[39]:


# predicting the data
prediction = lr.predict(x)
df1["predicted CTC"]=prediction
df1


# In[43]:


df1["predicted CTC"]


# In[72]:


sns.set(rc={"figure.figsize":(10,10)})
sns.lmplot(x= "Actual CTC", y="predicted CTC", data=df1, scatter_kws={"color":"yellow"}, line_kws={"color":"black"})
plt.show()


# In[73]:


#mean squres values
print("Mean square values:",metrics.mean_squared_error(y,prediction))


# In[76]:


# root mean square values
print("Root Mean Square Value:",np.sqrt(metrics.mean_squared_error(y,prediction)))


# In[77]:


# finally this is Predicted CTC
df1["predicted CTC"]


# In[ ]:




