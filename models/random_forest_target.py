#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix


# In[2]:


train = pd.read_csv('../data/new_train.csv')
test = pd.read_csv('../data/new_test.csv')
mcf = pd.read_csv('../data/monthly_card_features.csv')


# In[3]:


train = train.drop(columns=['Unnamed: 0', 'first_active_month']).set_index('card_id')
test = test.drop(columns=['Unnamed: 0', 'first_active_month']).set_index('card_id')


# In[4]:


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


# In[5]:


train.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in train.columns.values]
test.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in test.columns.values]


# In[6]:


X = train.drop(columns='target')
X.columns


# In[7]:


Y = train['target']


# In[8]:


X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.3)
X_val.shape


# In[9]:


rfr = RandomForestRegressor(max_depth=5, random_state=0, n_estimators=100)


# In[10]:


rfr.fit(X_train, Y_train)


# In[ ]:


print("Root Mean squared error: %.2f"
      % np.sqrt(mean_squared_error(Y_val, rfr.predict(X_val))))


# ## What if we bring in our chump-classifier?

# In[ ]:


train['chump'] = train['target'] < -20


# In[ ]:


chumps = train[train.chump]
nonchumps = train[~train.chump]


# In[ ]:


tr_chumps, tt_chumps = train_test_split(chumps, test_size=0.3)
tr_nonchumps, tt_nonchumps = train_test_split(nonchumps, test_size=0.3)


# In[ ]:


tr = pd.concat([tr_chumps, tr_nonchumps])
tt = pd.concat([tt_chumps, tt_nonchumps])
print('Training chumps:', sum(tr.chump)/len(tr))
print('Test chumps:    ', sum(tt.chump)/len(tt))


# In[ ]:


balanced = pd.concat([
    tr[tr.chump],
    tr[~tr.chump].sample(n=9 * len(tr[tr.chump]))
])
# Train on balanced data (10% chumps)
X = balanced.drop(columns=['chump', 'target'])
Y = balanced['chump']


# In[ ]:


print('Balanced chumps: ', sum(balanced.chump) / len(balanced))


# In[ ]:


clf = RandomForestClassifier(n_estimators=100, max_depth=7)
clf.fit(X, Y)


# ### Now retrain the regressor on only "non-chumps"

# In[ ]:


X = tr[~tr.chump].drop(columns=['chump', 'target'])
Y = tr[~tr.chump]['target']


# In[ ]:


Y.hist()


# In[ ]:


rfr = RandomForestRegressor(max_depth=5, random_state=0, n_estimators=100)
rfr.fit(X, Y)


# ## Apply trained models to full train set

# In[ ]:


X = tt.drop(columns=['chump', 'target'])
Y = tt['target']


# In[ ]:


X['pred_chump'] = clf.predict(X)


# In[ ]:


chumps = pd.DataFrame(X[X.pred_chump])
nonchumps = pd.DataFrame(X[~X.pred_chump])
chumps.head()


# In[ ]:


chumps['pred_target'] = -33.219281
chumps.head()


# In[ ]:


# X.loc[X.pred_chump, 'pred_target'] = -33.219281


# In[ ]:


X.head()


# In[ ]:


nonchumps['pred_target'] = rfr.predict(nonchumps.drop(columns='pred_chump'))
nonchumps.head()


# In[ ]:


# X.loc[~X.pred_chump, 'pred_target'] = rfr.predict(X[~X.pred_chump].drop(columns=['pred_chump', 'pred_target']))


# In[ ]:


X.head()


# In[ ]:


X = pd.concat([chumps, nonchumps])
X = X.join(Y)


# In[ ]:


np.sqrt(mean_squared_error(X.target, X.pred_target))


# In[ ]:


len(X[X.pred_chump])


# In[ ]:


sum(X.target < -20)


# In[ ]:


sum(X.pred_chump != (X.target < -20))


# In[ ]:


# Split dataset into tr for training and tt for final test
# tr and tt will maintain the same chump/nonchump ratio
train['chump'] = train['target'] < -20
chumps = train[train.chump]
nonchumps = train[~train.chump]
tr_chumps, tt_chumps = train_test_split(chumps, test_size=0.3)
tr_nonchumps, tt_nonchumps = train_test_split(nonchumps, test_size=0.3)
tr = pd.concat([tr_chumps, tr_nonchumps])
tt = pd.concat([tt_chumps, tt_nonchumps])

ranges = [1, 3, 5, 7, 10, 15, None]
models = [
    {'rmd': rmd,
     'rmf': rmf,
     'cmd': cmd,
     'cmf': cmf,
     'score': np.nan} 
    for rmd in ranges
    for rmf in ranges
    for cmd in ranges
    for cmf in ranges
]

for i, m in enumerate(models):
    # Train Classifier (rebalanced tr set)
    balanced = pd.concat([
        tr[tr.chump],
        tr[~tr.chump].sample(n=9 * len(tr[tr.chump]))
    ])
    # Train on balanced data (10% chumps)
    X = balanced.drop(columns=['chump', 'target'])
    Y = balanced['chump']
    clf = RandomForestClassifier(
        n_estimators=  100,
        max_depth=     m['cmd'],
        max_features=  m['cmf'],
        random_state=  0)
    clf.fit(X, Y)
    
    # Train Regressor (full tr set)
    X = tr[~tr.chump].drop(columns=['chump', 'target'])
    Y = tr[~tr.chump]['target']
    rfr = RandomForestRegressor(
        n_estimators=  100,
        max_depth=     m['rmd'],
        max_features=  m['rmf'],
        random_state=  0)
    rfr.fit(X, Y)
    
    # Apply models to tt holdout set
    X = tt.drop(columns=['chump', 'target'])
    Y = tt['target']
    X['pred_chump'] = clf.predict(X)
    chumps = pd.DataFrame(X[X.pred_chump])
    nonchumps = pd.DataFrame(X[~X.pred_chump])
    chumps['pred_target'] = -33.219281
    nonchumps['pred_target'] = rfr.predict(nonchumps.drop(columns='pred_chump'))
    X = pd.concat([chumps, nonchumps])
    X = X.join(Y)
    m['score'] = np.sqrt(mean_squared_error(X.target, X.pred_target))
    print(f'== {np.round((i + 1) / len(models) * 100, 2)}% ==\r', end='')


# In[ ]:


sorted(models, key = lambda x: x['score'])[:15]t


# In[23]:


train['chump'] = train['target'] < -20
# Train Classifier (rebalanced tr set)
balanced = pd.concat([
    train[train.chump],
    train[~train.chump].sample(n=9 * len(train[train.chump]))
])
# Train on balanced data (10% chumps)
X = balanced.drop(columns=['chump', 'target'])
Y = balanced['chump']
clf = RandomForestClassifier(
    n_estimators=  1000,
    max_depth=     3,
    max_features=  3,
    random_state=  0)
clf.fit(X, Y)

# Train Regressor (full tr set)
X = train[~train.chump].drop(columns=['chump', 'target'])
Y = train[~train.chump]['target']
rfr = RandomForestRegressor(
    n_estimators=  1000,
    max_depth=     10,
    max_features=  15,
    random_state=  0)
rfr.fit(X, Y)

# Apply models to tt holdout set
test['pred_chump'] = clf.predict(test)
chumps = pd.DataFrame(test[test.pred_chump])
nonchumps = pd.DataFrame(test[~test.pred_chump])
chumps['pred_target'] = -33.219281
nonchumps['pred_target'] = rfr.predict(nonchumps.drop(columns='pred_chump'))
test = pd.concat([chumps, nonchumps])
predictions = pd.DataFrame(test['pred_target'], index=test.index)
predictions.head()


# In[30]:


predictions = pd.DataFrame(test['pred_target'], index=test.index)
predictions.columns = ['target']
predictions.head()


# In[31]:


predictions.to_csv('../submission_stacked_rf.csv')


# # Forget the stacked model - tune RFR to full training data

# In[13]:


train['chump'] = train['target'] < -20
chumps = train[train.chump]
nonchumps = train[~train.chump]
tr_chumps, tt_chumps = train_test_split(chumps, test_size=0.3)
tr_nonchumps, tt_nonchumps = train_test_split(nonchumps, test_size=0.3)
tr = pd.concat([tr_chumps, tr_nonchumps]).drop(columns='chump')
tt = pd.concat([tt_chumps, tt_nonchumps]).drop(columns='chump')

ranges = [1, 3, 5, 7, 10, 15, None]
models = [
    {'rmd': rmd,
     'rmf': rmf,
     'score': np.nan} 
    for rmd in ranges
    for rmf in ranges
]

for i, m in enumerate(models):
    X = tr.drop(columns='target')
    Y = tr['target']
    rfr = RandomForestRegressor(
        n_estimators=  100,
        max_depth=     m['rmd'],
        max_features=  m['rmf'],
        random_state=  0)
    rfr.fit(X, Y)
    
    # Apply models to tt holdout set
    X = tt.drop(columns='target')
    Y = tt['target']
    X['pred_target'] = rfr.predict(X)
    X = X.join(Y)
    m['score'] = np.sqrt(mean_squared_error(X.target, X.pred_target))
    print(f'== {np.round((i + 1) / len(models) * 100, 2)}% ==\r', end='')


# In[14]:


sorted(models, key = lambda x: x['score'])[:15]


# In[15]:


test.columns


# ### Best model has `max_depth=10` and `max_features=15`

# In[11]:


X = train.drop(columns=['target', 'chump'])
Y = train['target']
rfr = RandomForestRegressor(
        n_estimators=  1000,
        max_depth=     10,
        max_features=  15,
        random_state=  0)
rfr.fit(X, Y)
pd.DataFrame(rfr.predict(test), index=test.index, columns='target').head()


# In[12]:


feature_imp = pd.Series(rfr.feature_importances_,index=X.columns).sort_values(ascending=False)
sns.barplot(x=feature_imp.head(10), y=feature_imp.index[:10])
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()


# In[18]:


predictions = pd.DataFrame(rfr.predict(test), index=test.index, columns=['target'])
predictions.head()


# In[19]:


predictions.to_csv('../submission.csv')


# In[19]:


print(
    np.mean(train[train.target > -20].target), ',',
    np.std(train[train.target > -20].target)
)


# In[14]:


train.columns


# In[ ]:




