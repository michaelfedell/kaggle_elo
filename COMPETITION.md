# Kaggle Competition

## Description

Imagine being hungry in an unfamiliar part of town and getting restaurant recommendations served up, based on your personal preferences, at just the right moment. The recommendation comes with an attached discount from your credit card provider for a local place around the corner!

Right now, Elo, one of the largest payment brands in Brazil, has built partnerships with merchants in order to offer promotions or discounts to cardholders. But do these promotions work for either the consumer or the merchant? Do customers enjoy their experience? Do merchants see repeat business? Personalization is key.

Elo has built machine learning models to understand the most important aspects and preferences in their customers’ lifecycle, from food to shopping. But so far none of them is specifically tailored for an individual or profile. This is where you come in.

In this competition, Kagglers will develop algorithms to identify and serve the most relevant opportunities to individuals, by uncovering signal in customer loyalty. Your input will improve customers’ lives and help Elo reduce unwanted campaigns, to create the right experience for customers.

## Data

### Format

The data is formatted as follows:

- train.csv and test.csv contain card_ids and information about the card itself - the first month the card was active, etc. train.csv also contains the target.
- historical_transactions.csv and new_merchant_transactions.csv are designed to be joined with train.csv, test.csv, and merchants.csv. They contain information about transactions for each card, as described above.
- merchants can be joined with the transaction sets to provide additional merchant-level information.

### Prediction Target

You are predicting a loyalty score for each card_id represented in test.csv and sample_submission.csv.

### File descriptions

- `train.csv` - the training set
- `test.csv` - the test set
- `sample_submission.csv` - a sample submission file in the correct format - contains all card_ids you are expected to predict for.
- `historical_transactions.csv` - up to 3 months' worth of historical transactions for each card_id
- `merchants.csv` - additional information about all merchants / merchant_ids in the dataset.
- `new_merchant_transactions.csv` - two months' worth of data for each card_id containing ALL purchases that card_id made at merchant_ids that were not visited in the historical data.

### Data fields

Data field descriptions are provided [here](./data/dictionary.xlsx).