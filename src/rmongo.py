import pandas as pd
import numpy as np
from pymongo import MongoClient
client = MongoClient('localhost', 27017)
db = client['class_db']
table = db['teachers']

df = pd.read_json('../data/data.json')
yes = df.iloc[0, :].to_dict()
for i in yes.keys():
    if type(yes[i]) == np.int64:
        yes[i] = int(yes[i])

# print(yes)
if __name__ == '__main__':
    example_record = yes #{'name':'moses', 'age':31, 'friends':['ted', 'gahl']}
    table.insert_one(example_record)
    # table.update_one({'name':'moses'}, {'$set':{'age':32}})
    table.find()
    # table.find().limit(1)
    # table.collection()
    for doc in table.find():
        print(doc)