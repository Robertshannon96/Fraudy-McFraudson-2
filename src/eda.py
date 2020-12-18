import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
import pickle

from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer

class DataPipeline:
    def __init__(self, df):
        self.df = df

    def add_fraud(self):
        """
        This function adds the 'fraud' column initially assigned to False. If the column 'acct_type
        has fraud contained anywhere in it, a wild card should pick that up and assign a True value to that row.
        """
        self.df['fraud'] = 0
        self.df['fraud'][self.df['acct_type'].str.contains('fraud(?!$)')] = 1

    def count_fraud(self):
        print(f'Total amount of fraudulent cases:', self.df['fraud'].sum())  # prints out total sum of fraud cases

    def drop_leaky(self):
        """
        Dropping leaky columns and columns not used in our models later on.
        """
        cols_to_drop = ['acct_type', 'approx_payout_date', 'gts', 'num_payouts', 'payout_type',
                        'sale_duration', 'sale_duration2', 'venue_name', 'venue_latitude',
                        'venue_longitude']
        for col in cols_to_drop:
            self.df.drop(col, axis=1, inplace=True)

    def clean(self):
        self.df['user_type'] = self.df['user_type'].replace(103, 6)  # reassigned 106 to 6 for better EDA
        for i in self.df.columns:
            self.df[i] = self.df[i].replace([np.nan], "-1")
        self.df['user_age'] = self.df['user_age'].astype(float)
        self.df['has_address'] = self.df['venue_address'].str.len() > 0  # Creates new column of binary value 0 or 1
        self.df['num_previous_payouts'] = self.df['previous_payouts'].apply(lambda x: len(x))
        self.df['num_ticket_types'] = self.df['ticket_types'].apply(lambda x: len(x))
        self.df['e_dom1'] = self.df['email_domain'].apply(lambda x: x.split(".")[0])
        self.df['e_dom2'] = self.df['email_domain'].apply(lambda x: ''.join(x.split(".")[1:]))
        self.df['listed'] = self.df['listed'].replace({'y': 1, 'n': 0})

    def nlp_vectorization(self):
        """
        Pulls out natural language features and vectorizes them with tf-idf
        vectorizers that have been fit to the full training dataset. Code
        that is commented out is for the generation and pickling of vectorizers
        left in incase of future modifications and tuning.

        Returns
        ----------
        text_df, name_df, org_df - Dense Panda dataframes with the results
                                    of vectorization
        """
        
        text = self.df['description'].apply(lambda text: BeautifulSoup(text, 'html.parser').get_text())
        # vecto = TfidfVectorizer(stop_words='english', max_features=5)
        # vecto.fit(text)
        infile = open('vectorizers/text_vec.pkl','rb')
        vecto = pickle.load(infile)
        text_vect = vecto.transform(text)
        text_df = pd.DataFrame(text_vect.todense())
        # f =  open('vectorizers/text_vec.pkl', 'wb')
        # pickle.dump(vecto, f)

        names = self.df['name']
        # name_vecto = TfidfVectorizer(stop_words='english', max_features=5)
        # name_vecto.fit(names)
        infile = open('vectorizers/name_vec.pkl','rb')
        name_vecto = pickle.load(infile)
        name_vect = name_vecto.transform(names)
        name_df = pd.DataFrame(name_vect.todense())
        # f =  open('vectorizers/name_vec.pkl', 'wb')
        # pickle.dump(name_vecto, f)

        org_desc = self.df['org_desc'].apply(lambda text: BeautifulSoup(text, 'html.parser').get_text())
        # org_vecto = TfidfVectorizer(stop_words='english', max_features=5)
        # org_vecto.fit(org_desc)
        infile = open('vectorizers/name_vec.pkl','rb')
        org_vecto = pickle.load(infile)
        org_vect = org_vecto.transform(org_desc)
        org_df = pd.DataFrame(org_vect.todense())
        # f =  open('vectorizers/org_vec.pkl', 'wb')
        # pickle.dump(org_vecto, f)

        return text_df, name_df, org_df
    
    def one_hot(self):
        """
        Pulls out the features that will actually be run through the model
        (beyond nlp) and one hot encodes the relevant features.

        Returns
        ----------
        df_one_hot - A pandas dataframe that when joined with the results
                        of nlp_vectorization can be run through the model.
        """

        variables = ['country', 'e_dom1', 'e_dom2', 'currency', 'venue_state', 'venue_country']
    
        X = self.df[['body_length', 'channels', 'delivery_method',
            'event_created', 'event_end', 'event_published',
            'event_start', 'fb_published', 'has_analytics', 'has_header', 
            'has_logo', 'listed', 'name_length', 'num_order',       
            'object_id', 'org_facebook', 'org_twitter',       
            'num_ticket_types', 'num_previous_payouts', 'show_map',       
            'user_age', 'user_created', 'user_type', 'has_address']]
        
        # df_one_hot = pd.get_dummies(X,columns=variables,drop_first=True)

        return X

    def format_input(self):
        self.clean()
        desc, name, org = self.nlp_vectorization()
        rest = self.one_hot()
        X = pd.concat([rest, desc, name, org], axis=1)
        return X

    def test_script_examples(self):
        examples = self.df.sample(1)
        examples.to_csv('../data/test_script_examples.csv')


class Eda:
    def __init__(self, df):
        self.df = df
        self.fraud = self.df[self.df['fraud'] == 1]
        self.not_fraud = self.df[self.df['fraud'] == 0]

    def comparison_hist(self):
        features = ['currency', 'user_age', 'user_type', 'channels', 'country']
        for feature in features:
            fig, ax = plt.subplots(2, figsize=(12, 8))
            fig.suptitle(f'{feature} in fraud vs not fraud')
            if type(self.fraud[feature]) == float:
                bins = 10
            else:
                bins = len(self.fraud[feature].unique())
            ax[0].hist(self.fraud[feature], edgecolor='black', lw=1.5, bins=bins)
            ax[0].set_title('Fraud Cases')
            ax[1].hist(self.not_fraud[feature], edgecolor='black', lw=1.5, bins=bins)
            ax[1].set_title('Not Fraud Cases')
            # ax[0].plt.xlabel(f'{feature}')
            # ax[0].plt.ylabel('count')
            # ax[1].plt.xlabel(f'{feature}')
            # ax[1].plt.ylabel('count')
            plt.savefig(f'../images/{feature}.png')


def main():
    df = pd.read_json('../data/data.json')
    df_pipe = DataPipeline(df)
    df_pipe.add_fraud()
    df_pipe.count_fraud()
    df_pipe.clean()
    df_pipe.test_script_examples()
    df_pipe.drop_leaky()
    df = df_pipe.df
    hist = Eda(df)
    hist.comparison_hist()


if __name__ == "__main__":
    main()
