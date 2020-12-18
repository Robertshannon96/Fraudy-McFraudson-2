import numpy as np
import pandas as pd
import pickle

from bs4 import BeautifulSoup
from r_eda import DataPipeline
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
        return self.model.predict_proba(inflow)

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

def get_example(filepath):
    ex = pd.read_json(filepath)
    pipe = DataPipeline(ex)
    example = pipe.format_input()
    return testing(example)
    
# DO NOT TOUCH THIS
def testing(example):
    """Takes an example code 
    Args:
        example ([type]): [description]
    """    
    clean_df = pd.read_csv('../data/test_script_examples.csv', index_col=0)
    # print(clean_df.head())
    clean_df2 = pd.DataFrame()
    for i in clean_df.columns:
        clean_df2.loc[0, i] = 0
    
    # clean_df = clean_df.replace(np.nan, 0)
    # clean_df = clean_df.fillna(0)

    # print(clean_df[clean_df != 0])
    for i in example.columns:
        # if i != np.nan:
        if i in clean_df2.columns:
            clean_df2[i] = example[i]
    return clean_df2
# DO NOT TOUCH THIS

def test_script_examples(df):
        clean_df = df.sample(1)
        for i in clean_df.columns:
            clean_df[i] = 0 
        clean_df.to_csv('../data/test_script_examples.csv')




if __name__ == '__main__':
    # X, y = get_data('../data/data.json')
   
    # test_script_examples(X)
    X_test = get_example('../data/example.json')
   
    
    # model = MyModel()
    # model.fit(X, y)
    # print(model.score(X, y))
    # f =  open('/Users/riley/Desktop/DSI/den-19/repos/Fraudy-McFraudson-2/src/model.pkl', 'wb')
    # pickle.dump(model, f)

    infile = open('model.pkl','rb')
    model = pickle.load(infile)
    # print(set(model.estimators_).difference(set(X_test.columns)))
    print(model.predict(X_test))