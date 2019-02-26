#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from functools import reduce


# In[2]:


trx = pd.read_csv('../data/historical_transactions.csv')
trx['new'] = False
merchants = pd.read_csv('../data/merchants.csv')
new_merchant_trx = pd.read_csv('../data/new_merchant_transactions.csv')
new_merchant_trx['new'] = True
train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')


# In[81]:


# What proportion of the training cards are in new merchant as well
len(set(train.card_id).intersection(set(new_merchant_trx.card_id)))    / len(train.card_id.unique())


# In[71]:


len(test.card_id.unique())


# Sample the card ID's to trim transaction data down to a unique set of 1000 cards

# In[3]:


random.seed(903)
sample = np.random.choice(trx.card_id.unique(), size=1000, replace=False)
trx[trx.card_id.isin(sample)].card_id.unique().size


# In[5]:


trx_sample = pd.concat([trx[trx.card_id.isin(sample)], new_merchant_trx[new_merchant_trx.card_id.isin(sample)]])
trx_sample.head()


# In[6]:


trx_sample.shape


# In[7]:


trx_sample.card_id.unique().size


# # EDA
# 
# Basic look at transactions

# In[85]:


trx.describe()


# In[8]:


trx.info()


# In[9]:


trx.head()


# In[10]:


print('auth flag: {}\n'
      'unique cards: {}\n'
      'cat_1: {}\n'
      'cat_2: {}\n'
      'cat_3: {}\n'
      'unique merch_cat: {}\n'
      'unique merch_id: {}\n'
      'state_id: {}\n'
      'subsector_id: {}\n'
      .format(
          trx.authorized_flag.unique(),
          len(trx.card_id.unique()),
          trx.category_1.unique(),
          trx.category_2.unique(),
          trx.category_3.unique(),
          len(trx.merchant_category_id.unique()),
          len(trx.merchant_id.unique()),
          len(trx.state_id.unique()),
          len(trx.subsector_id.unique())
      ), sep='\n')


# Some data cleaning/recoding first-steps:
# - `authorized_flag` to boolean
# - `city_id` to str
# - define `city_id` (to put on map)
# - `category_1` to boolean
# - `category_3` meaning for A, B, C?
# - units for `purchase_amount`?
# - `category_1` may need to exist as str
# - `purchase_date` to datetime

# In[11]:


trx.purchase_date = pd.to_datetime(trx.purchase_date)
trx.authorized_flag = trx.authorized_flag.apply(lambda x: True if x == 'Y' else False)
trx.city_id = trx.city_id.apply(str)


# In[12]:


merchants.info()


# Not going to worry about merchants for now. Will first focus on making predictions using only customer data.
# 
# First we will need to do some feature engineering for our customers

# ## Customer level feature engineering

# In[36]:


groups = trx.groupby('card_id')
last_day = trx.purchase_date.max()
users = pd.DataFrame()
users['tof'] = (last_day - groups.purchase_date.min()).apply(lambda x: x.days)
users['recency'] = (last_day - groups.purchase_date.max()).apply(lambda x: x.days)
users['frequency'] = groups.size()
users['log_freq'] = users.frequency.apply(np.log)
users['amt'] = groups.purchase_amount.sum()
users['log_amt'] = users['amt'].apply(np.log)
users['avg_amt'] = users['amt'] / users['frequency']
users['log_avg_amt'] = users['avg_amt'].apply(np.log)
users['charge_per_day'] = users['frequency'] / (users['tof'] + 1)
users['log_charge_per_day'] = users['charge_per_day'].apply(np.log)
users['max_amt'] = groups.purchase_amount.max()
users['log_max_amt'] =  users['max_amt'].apply(np.log)
users['n_declines'] = groups.size() - groups.authorized_flag.sum()
users['log_n_declines'] = users['n_declines'].apply(lambda x: np.log(x+1))


# In[10]:


users.log_n_declines.hist()


# In[37]:


users.head()


# It actually doesn't make sense to take logs on any features based on `purchase_amount` as it has been standardized and thus contains many negative values

# In[38]:


full = train.join(users, how='inner', on='card_id')
full.head()


# In[39]:


test_full = test.join(users, how='left', on='card_id')
test_full.shape


# In[40]:


# Log/standardize predictors
X = full[['feature_1', 'feature_2', 'feature_3', 'tof', 'recency', 'log_freq',
          'amt', 'avg_amt', 'max_amt', 'log_charge_per_day', 'n_declines']]
Y = full['target']


# In[42]:


test_X = test_full[['feature_1', 'feature_2', 'feature_3', 'tof', 'recency', 'log_freq',
          'amt', 'avg_amt', 'max_amt', 'log_charge_per_day', 'n_declines']]
test_X.shape
test_X[test_X.isna().any(axis=1)].size


# In[43]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
len(X_train.dropna())


# ## Linear Regression

# In[44]:


# Create linear regression object
regr = LinearRegression()

# Train the model using the training sets
regr.fit(X_train, Y_train)

# Make predictions using the testing set
Y_pred = regr.predict(X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Root Mean squared error: %.2f"
      % np.sqrt(mean_squared_error(Y_test, Y_pred)))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(Y_test, Y_pred))


# In[45]:


test_preds = regr.predict(test_X)
test_preds.shape


# In[78]:


# Format predictions for submission
pd.DataFrame({'card_id':test['card_id'], 'target':test_preds})    .to_csv('./submissions.csv', index=False)


# ## Alternate User-Month Analysis

# In[18]:


trx_all = pd.concat([trx, new_merchant_trx], axis=0).reset_index()
trx_all.shape


# In[24]:


trx_all.columns


# In[8]:


trx_sample.columns


# In[9]:


trx_sample.head()


# In[6]:


groups = trx_sample.groupby(['card_id', 'month_lag'])


# In[35]:


trx_sample.sort_values(['card_id', 'month_lag']).drop_duplicates(['card_id', 'merchant_id'])    .groupby(['card_id', 'month_lag']).merchant_id.size().unstack().fillna(0).cumsum(axis=1)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




