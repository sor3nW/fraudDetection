import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#reading the data
credit_card_data = pd.read_csv('/Users/sorenw/Desktop/projects/CreditCardFraudDetecor/creditcard.csv')
credit_card_data.head()
inf = credit_card_data.info()

print(credit_card_data['Class'].value_counts())
# result 0    284315
#        1    492
# this dataset is highly unbalanced 
# 0->normal transaction
# 1->fraud transaction

#seperating the data for analysis
regular = credit_card_data[credit_card_data['Class']==0]
fraud = credit_card_data[credit_card_data['Class']==1]

print(regular.shape)
print(fraud.shape)

#statistical measures of the data
print(regular.Amount.describe())
print(fraud.Amount.describe())

#compare the values for both transaction types

print(credit_card_data.groupby('Class').mean())

#under sampling
regular_sample = regular.sample(n=492)

#concatenating two dataframes
new_dataset = pd.concat([regular_sample, fraud], axis=0)

