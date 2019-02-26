#!/usr/bin/env python
# coding: utf-8

# In[186]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix


# In[73]:


train = pd.read_csv('../data/new_train.csv')
test = pd.read_csv('../data/new_test.csv')
mcf = pd.read_csv('../data/monthly_card_features.csv')


# In[74]:


train = train.drop(columns=['Unnamed: 0', 'first_active_month']).set_index('card_id')
test = test.drop(columns=['Unnamed: 0', 'first_active_month']).set_index('card_id')


# In[75]:


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


# In[76]:


train.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in train.columns.values]
test.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in test.columns.values]


# In[77]:


train['chump'] = train['target'] < -20


# In[78]:


train.columns


# In[141]:


balanced = pd.concat([
    train[train.chump],
    train[~train.chump].sample(n=9 * len(train[train.chump]))
])
balanced.chump.describe()


# In[176]:


# Train on balanced data (10% chumps)
X = balanced.drop(columns=['chump', 'target'])
Y = balanced['chump']

# Train on original, imbalanced data (1% chumps)
# X = train.drop(columns=['chump', 'target'])
# Y = train['chump']


# In[177]:


X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.3)
X_train.shape


# In[178]:


X_val.shape


# In[179]:


clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_val)
print("Accuracy:", accuracy_score(Y_val, Y_pred))


# In[180]:


accuracy_score(train['chump'], clf.predict(train.drop(columns=['chump', 'target'])))


# In[181]:


1 - sum(train['target'] < -20) / len(train)


# In[182]:


feature_imp = pd.Series(clf.feature_importances_,index=X_train.columns).sort_values(ascending=False)
feature_imp


# In[198]:


confusion_matrix(Y_val, Y_pred)


# In[195]:


sns.barplot(x=feature_imp.head(10), y=feature_imp.index[:10])
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()


# In[1]:


train


# In[ ]:




