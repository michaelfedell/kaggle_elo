# Elo Kaggle Competition

Elo Merchant Category Recommendation Competition on [Kaggle](https://www.kaggle.com/c/elo-merchant-category-recommendation) - Team Captain Krebs

## Overview

TODO: Update Overview

More information can be found [here](./COMPETITION.md)

## Getting Started

TODO: Update Getting Started

## Data

The raw data files are quite large and thus will not be included as part of this repo. This project will assume that all data files will be downloaded from Kaggle [here](https://www.kaggle.com/c/elo-merchant-category-recommendation/data) and kept in the `./data/` folder.

At a glance, the available data describe customer's credit cards and the purchases they have made over several months. More description about the data can be found [here](./COMPETITION.md#Data)

Attribute | Description
--- | ---
merchant_id	| Unique merchant identifier
merchant_group_id | Merchant group (anonymized )
merchant_category_id | Unique identifier for merchant category (anonymized )
subsector_id | Merchant category group (anonymized )
numerical_1 | anonymized measure
numerical_2 | anonymized measure
category_1 | anonymized category
most_recent_sales_range | Range of revenue (monetary units) in last active month --> A > B > C > D > E
most_recent_purchases_range | Range of quantity of transactions in last active month --> A > B > C > D > E
avg_sales_lag3 | Monthly average of revenue in last 3 months divided by revenue in last active month
avg_purchases_lag3 | Monthly average of transactions in last 3 months divided by transactions in last active month
active_months_lag3 | Quantity of active months within last 3 months
avg_sales_lag6 | Monthly average of revenue in last 6 months divided by revenue in last active month
avg_purchases_lag6 | Monthly average of transactions in last 6 months divided by transactions in last active month
active_months_lag6 | Quantity of active months within last 6 months
avg_sales_lag12 | Monthly average of revenue in last 12 months divided by revenue in last active month
avg_purchases_lag12 | Monthly average of transactions in last 12 months divided by transactions in last active month
active_months_lag12 | Quantity of active months within last 12 months
category_4 | anonymized category
city_id | City identifier (anonymized )
state_id | State identifier (anonymized )
category_2 | anonymized category

## Collaborators

- [Sabarish Chockalingam](https://github.com/sabarishchockalingam2017)
- [Joseph (JD) Cook](https://github.com/josephd8)
- [Ruixiang (James) Fan](https://github.com/rfq4587)
- [Micahel Fedell](https://github.com/michaelfedell)
- [Henry Park](https://github.com/henrypark133)