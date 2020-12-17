import numpy as np
import pandas as pd
import pickle

from bs4 import BeautifulSoup
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

class MyModel():

    def __init__(self):
        self.model = RandomForestClassifier()

    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, inflow):
        return self.model.predict(inflow)

    def score(self):
        pass

def get_data(filepath):
    df = pd.read_json(filepath)

    df['fraud'] = False
    df['fraud'][df['acct_type'].str.contains('fraud(?!$)')] = True
    y = df['fraud']

    text = df['description'].apply(lambda text: BeautifulSoup(text, 'html.parser').get_text())
    vecto = TfidfVectorizer(stop_words='english', max_features=50)
    vecto.fit(text)
    text_vect = vecto.transform(text)
    text_df = pd.DataFrame(text_vect.todense())

    df['num_previous_payouts'] = df['previous_payouts'].apply(lambda x: len(x))
    df['e_dom1'] = df['email_domain'].apply(lambda x: x.split(".")[0])
    df['e_dom2'] =  df['email_domain'].apply(lambda x: ''.join(x.split(".")[1:]))  
    df['has_address'] = df['venue_address'].str.len() > 0 

    variable = ['country', 'e_dom1', 'e_dom2']
    df_one_hot = pd.get_dummies(df,columns=variable,drop_first=True)

    # X = df[['body_length', 'channels', 'currency', 'delivery_method',
    #     'event_created', 'event_end', 'event_published',
    #     'event_start', 'fb_published', 'has_analytics', 'has_header', 
    #     'has_logo', 'listed', 'name', 'name_length', 'num_order',       
    #     'object_id', 'org_desc', 'org_facebook', 'org_name', 'org_twitter',       
    #     'payee_name', 'num_previous_payouts', 'show_map', 'ticket_types',       
    #     'user_age', 'user_created', 'user_type', 'has_address', 
    #     'venue_country', 'venue_state', 'fraud']]
    X = df[['gts']]
    pd.concat([X, text_df, df_one_hot], axis=1)

    return X, y

if __name__ == '__main__':
    X, y = get_data('data/data.json')
    model = MyModel()
    model.fit(X, y)
    f =  open('model.pkl', 'wb')
    pickle.dump(model, f)