import numpy as np
import pandas as pd
import pickle

from bs4 import BeautifulSoup
from eda import DataPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

pd.set_option('display.max_columns', None)

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
    pipe = DataPipeline(df)
    pipe.add_fraud()
    pipe.drop_leaky()
    pipe.clean()

    y = pipe.df['fraud']

    text = df['description'].apply(lambda text: BeautifulSoup(text, 'html.parser').get_text())
    vecto = TfidfVectorizer(stop_words='english', max_features=5)
    text_vect = vecto.fit_transform(text)
    text_df = pd.DataFrame(text_vect.todense())
    # print(text_df.head())

    names = df['name']
    name_vecto = TfidfVectorizer(stop_words='english', max_features=5)
    name_vect = name_vecto.fit_transform(names)
    name_df = pd.DataFrame(name_vect.todense())

    org_desc = df['org_desc'].apply(lambda text: BeautifulSoup(text, 'html.parser').get_text())
    org_vecto = TfidfVectorizer(stop_words='english', max_features=5)
    org_vect = org_vecto.fit_transform(org_desc)
    org_df = pd.DataFrame(org_vect.todense())

    variables = ['country', 'e_dom1', 'e_dom2', 'currency', 'venue_state', 'venue_country']
    
    X = pipe.df[['body_length', 'channels', 'delivery_method',
        'event_created', 'event_end', 'event_published',
        'event_start', 'fb_published', 'has_analytics', 'has_header', 
        'has_logo', 'listed', 'name_length', 'num_order',       
        'object_id', 'org_facebook', 'org_twitter',       
        'num_ticket_types', 'num_previous_payouts', 'show_map',       
        'user_age', 'user_created', 'user_type', 'has_address', 
        'country', 'e_dom1', 'e_dom2', 'currency', 'venue_state', 'venue_country']]
    
    df_one_hot = pd.get_dummies(X,columns=variables,drop_first=True)
    
    X = pd.concat([text_df, name_df, org_df, df_one_hot], axis=1)
    # print(X.head(1))
    return X, y

if __name__ == '__main__':
    X, y = get_data('data/data.json')
    model = MyModel()
    model.fit(X, y)
    f =  open('model.pkl', 'wb')
    pickle.dump(model, f)