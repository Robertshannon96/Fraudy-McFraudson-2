import numpy as np
import pandas as pd
import pickle

from bs4 import BeautifulSoup
from eda import DataPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

pd.set_option('display.max_columns', None)

class MyModel():
    """
    A minimal class to house a supervised learning model.
    """
    
    def __init__(self):
        self.model = RandomForestClassifier()

    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, inflow):
        return self.model.predict(inflow)

    def score(self, X, y):
        return self.model.score(X, y)

def get_data(filepath):
    """
    Uses the processes of the DataPipeline class to pull and clean data for
    fitting a model.

    Parameters
    ----------
    filepath - The filepath to the raw data.

    Returns
    ----------
    X, y - The features and targets for model fitting.
    """

    df = pd.read_json(filepath)
    pipe = DataPipeline(df)
    pipe.add_fraud()
    pipe.drop_leaky()
    pipe.clean()

    y = pipe.df['fraud']
    
    desc, name, org = pipe.nlp_vectorization()
    rest = pipe.one_hot()
    X = pd.concat([rest, desc, name, org], axis=1)
    
    return X, y

if __name__ == '__main__':
    X, y = get_data('data/data.json')
    model = MyModel()
    model.fit(X, y)
    print(model.score(X, y))
    f =  open('model.pkl', 'wb')
    pickle.dump(model, f)