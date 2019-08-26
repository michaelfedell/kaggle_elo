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


# In[3]:


trx.purchase_date = pd.to_datetime(trx.purchase_date)
trx.authorized_flag = trx.authorized_flag.apply(lambda x: True if x == 'Y' else False)
trx.city_id = trx.city_id.apply(str)


# In[4]:


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


# In[5]:


# Create linear regression object
lin_reg = LinearRegression()

# Train the model using the training sets
lin_reg.fit(X_train, Y_train)

# Make predictions using the testing set
Y_pred = lin_reg.predict(X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Root Mean squared error: %.2f"
      % np.sqrt(mean_squared_error(Y_test, Y_pred)))
print('R-Sq: %.2f' % r2_score(Y_test, Y_pred))


# In[ ]:


test_preds = regr.predict(test_X)
test_preds.shape


# In[ ]:


# Format predictions for submission
pd.DataFrame({'card_id':test['card_id'], 'target':test_preds})#     .to_csv('./submissions.csv', index=False)


# In[ ]:




