#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss
from sklearn.metrics import mean_squared_error
from math import sqrt
import lightgbm as lgb
import time


# In[2]:


train = pd.read_csv(os.path.join('data', 'train.csv'))
test = pd.read_csv(os.path.join('data', 'test.csv'))
hist = pd.read_csv(os.path.join('data', 'historical_transactions.csv'))
merch = pd.read_csv(os.path.join('data', 'merchants.csv'))
new_merch = pd.read_csv(os.path.join('data', 'new_merchant_transactions.csv'))
mcf = pd.read_csv("monthly_card_features.csv")
new_train = pd.read_csv("new_train.csv")
new_test = pd.read_csv("new_test.csv")


# In[3]:


hist["purchase_raw"] = hist["purchase_amount"] / 0.00150265118 + 497.06
new_merch["purchase_raw"] = new_merch["purchase_amount"] / 0.00150265118 + 497.06
train['target_raw'] = 2**train['target']


# In[151]:


#mcf.groupby(["card_id"]).mean().merge(train, on = "card_id").corr()


# In[4]:


temp_mcf = mcf.groupby(["card_id"]).mean()


# In[5]:


train = train.merge(temp_mcf, on = "card_id")


# In[6]:


test = test.merge(temp_mcf , on = "card_id")


# In[7]:


tr = new_train.merge(temp_mcf, on = "card_id")


# In[8]:


te = new_test.merge(temp_mcf, on = "card_id")


# In[9]:


training = tr.drop([tr.columns[0],"first_active_month", "card_id", "target"], axis = 1)


# In[10]:


testing = te.drop([te.columns[0],"first_active_month", "card_id"], axis = 1)


# In[13]:


target = tr["target"]


# In[14]:


tr_no = tr[tr['target'] > -20]
training_no = tr_no.drop([tr_no.columns[0],"first_active_month", "card_id", "target"], axis = 1)
target_no = tr_no['target']


# In[15]:


training_no.head()
training_no.describe()


# ## Cross Validation

# In[251]:





# In[16]:


training.shape


# In[17]:


training = np.array(training)
testing = np.array(testing)


# In[17]:


#params = {"n_estimators":[25,30],"learning_rate":[.01, .1], "max_depth":[5,8]}


# In[18]:


start_time = time.time()

params = {"n_estimators":[100, 200, 300],"learning_rate":[.01, .1, 1], "max_depth":[5, 8, 10]}
gsearch = GridSearchCV(estimator = GradientBoostingRegressor(n_estimators=100,learning_rate=0.01,
                                                               max_depth=5,random_state=0), 
param_grid = params, scoring='r2',n_jobs=4,iid=False, cv=5)
out = gsearch.fit(training,target)

end_time = time.time()


# In[19]:


out


# In[20]:


out.best_estimator_


# In[21]:


out.grid_scores_, out.best_params_, out.best_score_


# In[25]:


(end_time - start_time) / 60


# In[31]:





# In[ ]:





# In[22]:


# kf = KFold(n_splits=2)
# kf.get_n_splits(training)


# In[23]:


# rmse = []

# parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
# svc = svm.SVC(gamma="scale")
# clf = GridSearchCV(svc, parameters, cv=5)
# clf.fit(iris.data, iris.target)

# for train_index, test_index in kf.split(training_no):
#     print("TRAIN:", train_index, "TEST:", test_index)
#     X_train, X_test = training[train_index], training[test_index]
#     y_train, y_test = target[train_index], target[test_index]
    
#     for i in range(10):
#     est = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, 
#                                 max_depth=10, random_state=0, loss='ls').fit(X_train, y_train)
    
#     preds = est.predict(X_test)
#     rms = sqrt(mean_squared_error(y_test, preds))
#     print(rms)
#     rmse.append(rms)
    


# In[221]:





# In[ ]:





# In[26]:


est = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, 
                                max_depth=5, random_state=0, loss='ls').fit(training, target)


# In[27]:


preds = est.predict(testing)


# In[32]:


est.feature_importances_


# In[51]:


nam = list(tr.columns[3:])


# In[54]:


del nam[3:4]


# In[57]:


importance = pd.DataFrame(
{"Feature" : nam,
"Importance" : est.feature_importances_}
)


# In[63]:


importance.sort_values("Importance", ascending=False).to_clipboard(excel = True)


# In[ ]:





# In[74]:





# In[ ]:





# In[ ]:





# In[28]:


submit = pd.DataFrame(
{"card_id" : te["card_id"],
"target" : preds}
)


# In[29]:


submit.to_csv("submission4.csv", index = False)
submit.head()


# In[30]:


submit.shape


# In[ ]:





# In[260]:





# In[ ]:





# In[ ]:





# ## LightGBM

# In[ ]:





# In[266]:


outs = tr[tr.outlier]
ins = tr[-tr.outlier]


# In[ ]:





# In[286]:


tra = tr
tes = te
tra['outlier'] = tra['target'] < -20


# In[287]:


target = tra['outlier']
del tra['outlier']
del tra['target']


# In[288]:


features = [c for c in tra.columns if c not in ['Unnamed: 0', 'card_id', 'first_active_month']]
categorical_feats = [c for c in features if 'feature_' in c]


# In[289]:


param = {'num_leaves': 31,
         'min_data_in_leaf': 30, 
         'objective':'binary',
         'max_depth': 6,
         'learning_rate': 0.01,
         "boosting": "rf",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9 ,
         "bagging_seed": 11,
         "metric": 'binary_logloss',
         "lambda_l1": 0.1,
         "verbosity": -1,
         "random_state": 2333}


# In[291]:


get_ipython().run_cell_magic('time', '', 'folds = KFold(n_splits=2, shuffle=True, random_state=15)\noof = np.zeros(len(tra))\npredictions = np.zeros(len(tes))\nfeature_importance_df = pd.DataFrame()\n\nstart = time.time()\n\n\nfor fold_, (trn_idx, val_idx) in enumerate(folds.split(tra.values, target.values)):\n    print("fold n°{}".format(fold_))\n    trn_data = lgb.Dataset(tra.iloc[trn_idx][features], label=target.iloc[trn_idx], categorical_feature=categorical_feats)\n    val_data = lgb.Dataset(tra.iloc[val_idx][features], label=target.iloc[val_idx], categorical_feature=categorical_feats)\n\n    num_round = 10000\n    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=100, early_stopping_rounds = 200)\n    oof[val_idx] = clf.predict(tra.iloc[val_idx][features], num_iteration=clf.best_iteration)\n    \n    fold_importance_df = pd.DataFrame()\n    fold_importance_df["feature"] = features\n    fold_importance_df["importance"] = clf.feature_importance()\n    fold_importance_df["fold"] = fold_ + 1\n    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)\n    \n    predictions += clf.predict(tes[features], num_iteration=clf.best_iteration) / folds.n_splits\n\nprint("CV score: {:<8.5f}".format(log_loss(target, oof)))')


# In[302]:


trn_data = lgb.Dataset(tr[features], label=target, categorical_feature=categorical_feats)


# In[303]:


clf = lgb.train(param, trn_data, 100)


# In[308]:


tes = tes.drop(["Unnamed: 0", "first_active_month", "card_id"], axis = 1)


# In[324]:


y_pred=clf.predict(tes)


# In[326]:


pd.DataFrame(y_pred).hist()


# In[322]:


for i in range(0,len(y_pred)):
    if y_pred[i]>=.2:       # setting threshold to .5
        y_pred[i]=1
    else:  
        y_pred[i]=0


# In[323]:


sum(y_pred == 1)


# In[304]:


#Prediction
y_pred=clf.predict(tes)
#convert into binary values
for i in range(0,99):
    if y_pred[i]>=.5:       # setting threshold to .5
        y_pred[i]=1
    else:  
       y_pred[i]=0


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




