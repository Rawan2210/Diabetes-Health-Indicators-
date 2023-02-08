#!/usr/bin/env python
# coding: utf-8

# In[7]:


import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
random.seed(1)


# In[8]:


#read in the dataset (select 2015)
df = pd.read_csv("diabetes_012_health_indicators_BRFSS2015.csv")
df.head()


# In[9]:


shape = df.shape
print(f"This Data Frame have {shape[0]} rows and {shape[1]} columns")


# In[10]:


df.info()


# In[11]:


#check missing value
df.isnull().sum()


# In[12]:


df.isnull().sum().sum()


# In[13]:


#check that the data loaded in is in the correct format
pd.set_option('display.max_columns', 500)
df.head()


# In[14]:


#Drop Missing Values - knocks 100,000 rows out right awa
df=df.dropna()
df.shape


# In[15]:


#Check how many respondents have no diabetes, prediabetes or diabetes. Note the class imbalance!
df.groupby(['Diabetes_012']).size()


# In[16]:


df['Diabetes_012'].value_counts(normalize=True).plot(kind='bar');
#0 is for no diabetes or only during pregnancy, 1 is for pre-diabetes or borderline diabetes, 2 is for yes diabetes


# we observe that less than 20% of the patient are with diabetes

# In[17]:


df['Sex'].value_counts().plot(kind='bar');
#Male is 1 and female is 0 


# we have more female than male.

# In[18]:


sns.heatmap(pd.crosstab(df['Diabetes_012'], df['Sex']), annot=True);


# In[19]:


df['HighChol'].value_counts(normalize=True).plot(kind='bar')


# About 45% of the Respondents have been told they have high cholestero

# In[20]:


sns.countplot(x='Education', hue='Diabetes_012', data=df);


# In[21]:


plt.hist(df['BMI'], bins=10);


# A lot of Respondent BMI falls between the 20-30 range

# In[22]:


df['HighBP'].value_counts(normalize=True).plot(kind='bar');


# About 45% of the Respondents have been told they have high blood pressure.

# In[23]:


plt.figure(figsize = (20,10))
sns.heatmap(df.corr(),annot=True , cmap ='coolwarm' )


# In[24]:


x= df.corr()


# In[25]:


col = x.columns
col


# In[36]:


X = df[['HighBP', 'HighChol', 'CholCheck', 'Smoker',
       'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
       'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
       'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income',
       'BMI']]
y = df[["Diabetes_012"]]


# In[37]:


y


# In[38]:


y["Diabetes_012"].value_counts()


# In[42]:


diff_class = (y[y['Diabetes_012']==1].shape[0]  / y[y['Diabetes_012']==0].shape[0])*100
print(f"diabetes class is {diff_class}% of total df")


# In[44]:


sns.countplot(y["Diabetes_012"])
labels = ["no diabetes", "diabetes"]
plt.title("No Diabetes VS Diabetes")
plt.show()


# In[45]:


#import relevant libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# In[46]:


#features extraction
x = df[['HighBP', 'HighChol', 'HeartDiseaseorAttack',
       'PhysActivity', 'GenHlth', 'PhysHlth', 'DiffWalk', 'Age', 'Education',
       'Income', 'BMI']]
y = df['Diabetes_012']


# In[47]:


x_train , x_test , y_train , y_test = train_test_split(x,y, test_size=0.3 , random_state=1)


# In[48]:


logreg = LogisticRegression() 
logreg.fit(x_train, y_train) 
y_pred = logreg.predict(x_test) 
print("Accuracy={:.4f}".format(logreg.score(x_test,y_test))) 


# In[49]:


y_pred=logreg.predict(x_test)

print('training set score: {:.4f}'.format(logreg.score(x_train, y_train)))

print('test set score: {:.4f}'.format(logreg.score(x_test, y_test)))


# In[50]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[ ]:




