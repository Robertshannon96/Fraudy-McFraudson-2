import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd


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
