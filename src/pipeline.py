from pandas_profiling import ProfileReport
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from pymongo import MongoClient
client = MongoClient('localhost', 27017)
db = client['creditcard']
table = db['fraudster']

pd.set_option('display.max_columns', None)

if __name__ == '__main__':
    df = pd.read_json('../data/data.json')
    df['fraud'] = False
    df['fraud'][df['acct_type'].str.contains('fraud(?!$)')] = True
    # df = df.dropna()
    df['has_address'] = df['venue_address'].str.len() > 0
    not_fraud = df.loc[df['fraud'] == 0]
    fraud = df.loc[df['fraud'] == 1]
    
    table.collection.insert_many(df.to_dict('dict'))
    
    # df = df.dropna()
    # print(not_fraud.head())
    
    # print(df.head())
    # print(df['fraud'].sum())
    # eda_report = ProfileReport(df)
    # eda_report.to_file(output_file='../html/cc_eda.html')
    # not_fraud.fillna('venue_country', 'None')
    # fraud.fillna('venue_country', 'None')
    # fig, ax = plt.subplots(2, figsize=(12,8))
    # fig.suptitle('Interest rates in fraud vs not fraud')
    # ax[0].hist(fraud['has_address'], edgecolor='black', lw=1.5, bins=len(fraud['country'].unique()))
    # ax[0].set_title('Fraud Cases')
    # ax[1].hist(not_fraud['has_address'], edgecolor='black', lw=1.5, bins=len(fraud['country'].unique()))
    # ax[1].set_title('Not Fraud Cases')

    # plt.show()
    # df['has_address'] = False
    
    # print(type(df['venue_address'][1]))
    # print(df['venue_address'][1])
    # print(type(df['has_address'][1]))
    # print(df['has_address'][1])

    # print(not_fraud.head(300))