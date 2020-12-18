import pandas as pd
import numpy as np
from pymongo import MongoClient
client = MongoClient('localhost', 27017)
db = client['class_db']
table = db['test']

def whats_my_risk(x):
    if x > .6:
        return 'High'
    elif x > .3:
        return 'Medium'
    else:
        return 'Low'

df = pd.read_json('../data/data.json')
yes = df.iloc[0, :].to_dict()
for i in yes.keys():
    if type(yes[i]) == np.int64:
        yes[i] = int(yes[i])

yes['FRAUD_RISK'] = whats_my_risk(.2)

# print(yes)
if __name__ == '__main__':
    example_record = yes #{'name':'moses', 'age':31, 'friends':['ted', 'gahl']}
    table.update_one({'user_created': 1259613950}, {'$set':{'FRAUD':'MEGA HIGH'}})
    table.insert_one(example_record)

    table.find()
    # table.find().limit(1)
    # table.collection()
    for doc in table.find():
        print()
        print(doc)