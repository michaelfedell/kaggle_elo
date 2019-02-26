#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PyNomaly import loop
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN


# In[6]:


train = pd.read_csv('../data/new_train.csv')
test = pd.read_csv('../data/new_test.csv')
mcf = pd.read_csv('../data/monthly_card_features.csv')
train = train.drop(columns=['Unnamed: 0', 'first_active_month']).set_index('card_id')
test = test.drop(columns=['Unnamed: 0', 'first_active_month']).set_index('card_id')
train = train.join(mcf.groupby('card_id').agg({
    'amt_total': [np.min, np.mean, np.max],
    'NDR':       [np.min, np.mean, np.max],
    'n_new_merchants':   np.mean,
    'n_total_merchants': np.max
}))
test = test.join(mcf.groupby('card_id').agg({
    'amt_total': [np.min, np.mean, np.max],
    'NDR':       [np.min, np.mean, np.max],
    'n_new_merchants':   np.mean,
    'n_total_merchants': np.max
}))
train.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in train.columns.values]
test.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in test.columns.values]


# In[4]:


for col in train.columns:
    print(col)
    train[col].hist()
    plt.show()


# In[3]:


scaler = StandardScaler()
scaler.fit(train)


# In[4]:


scaler.transform(train)


# In[5]:


train = pd.DataFrame(data=scaler.transform(train),columns = train.columns,index=train.index)


# In[6]:


train.columns


# In[7]:


train = train.drop(columns=['feature_1', 'feature_2', 'feature_3', 'target'])


# In[ ]:


db = DBSCAN(eps=0.6, min_samples=50).fit(train)


# In[21]:


m = loop.LocalOutlierProbability(train, extent=2, n_neighbors=20, cluster_labels=list(db.labels_)).fit()


# In[ ]:


scores = m.local_outlier_probabilities
scores.head()


# In[8]:


m = loop.LocalOutlierProbability(train).fit()
scores = m.local_outlier_probabilities
print(scores) 


# In[9]:


train.to_csv('temp.csv')


# In[4]:


tmp = pd.read_csv('../tmp.csv')


# In[5]:


tmp.head()


# In[9]:


tmp = tmp.set_index('card_id').join(train['target'])


# In[13]:


chump_ids = tmp[tmp.target < -20].index.values
pred_chump_ids = tmp[tmp.loop > 0.5].index.values


# In[18]:


len(set(chump_ids).intersection(pred_chump_ids)) / len(chump_ids)


# In[31]:





# In[ ]:





# In[ ]:




