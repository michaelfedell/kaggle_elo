#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 10:05:39 2019

@author: michaelfedell
"""

import pandas as pd
import numpy as np
from os import path

# Load all raw data
historical_trx = pd.read_csv(path.join('data', 'historical_transactions.csv'))
merchants = pd.read_csv(path.join('data', 'merchants.csv'))
new_merchant_trx = pd.read_csv(path.join('data', 'new_merchant_transactions.csv'))
train = pd.read_csv(path.join('data', 'train.csv'))
test = pd.read_csv(path.join('data', 'test.csv'))

# Combine new merchant and historical transactions
historical_trx['new'] = False
new_merchant_trx['new'] = True
trx = pd.concat([historical_trx, new_merchant_trx])

# Transform datetime flag and id's to appropriate dtypes
trx.purchase_date = pd.to_datetime(trx.purchase_date)
trx.authorized_flag = trx.authorized_flag.apply(lambda x: True if x == 'Y' else False)
trx.city_id = trx.city_id.apply(str)


def safe_log(x):
    if x <= 0:
        return 0
    else:
        return np.log(x)

last_day = trx.purchase_date.max()
groups = trx.groupby('card_id')
users = pd.DataFrame()
users['tof'] = (last_day - groups.purchase_date.min()).apply(lambda x: x.days)
users['recency'] = (last_day - groups.purchase_date.max()).apply(lambda x: x.days)
users['frequency'] = groups.size()
users['log_freq'] = users.frequency.apply(np.log)
users['amt'] = groups.purchase_amount.sum()
users['avg_amt'] = users['amt'] / users['frequency']
users['charge_per_day'] = users['frequency'] / (users['tof'] + 1)
users['log_charge_per_day'] = users['charge_per_day'].apply(np.log)
users['max_amt'] = groups.purchase_amount.max()
users['n_declines'] = groups.size() - groups.authorized_flag.sum()
users['log_n_declines'] = users['n_declines'].apply(safe_log)
users['prop_new'] = groups.new.sum() / groups.size()
users['merch_cat_1_Y'] = groups.apply(lambda x: (x['category_1'] == 'Y').sum())
#users['merch_cat_4_Y'] = groups.apply(lambda x: (x['category_4'] == 'Y').sum())
users['merch_cat_2_1'] = groups.apply(lambda x: (x['category_2'] == 1).sum())
users['merch_cat_2_2'] = groups.apply(lambda x: (x['category_2'] == 2).sum())
users['merch_cat_2_3'] = groups.apply(lambda x: (x['category_2'] == 3).sum())
users['merch_cat_2_4'] = groups.apply(lambda x: (x['category_2'] == 4).sum())
users['merch_cat_2_5'] = groups.apply(lambda x: (x['category_2'] == 5).sum())


full = train.join(users, how='inner', on='card_id')
test_full = test.join(users, how='left', on='card_id')
