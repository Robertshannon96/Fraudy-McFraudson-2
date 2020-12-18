import numpy as np
import pandas as pd
import pickle

from bs4 import BeautifulSoup
from r_eda import DataPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn import (
    cluster, datasets,
    decomposition, ensemble, manifold,
    random_projection, preprocessing)

from sklearn.metrics import f1_score

pd.set_option('display.max_columns', None)

class MyModel():
    """
    A minimal class to house a supervised learning model.
    """
    
    def __init__(self):
        self.model = RandomForestClassifier(random_state=69)

    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, inflow):
        return self.model.predict_proba(inflow)
        # return self.model.predict(inflow)

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

def inflow_channel(filepath):
    pass
    
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

def test_inflow_channel(filepath):
    ex = pd.DataFrame(pd.read_json(filepath).iloc[0, :]).T
    print(ex.shape)
    pipe = DataPipeline(ex)
    example = pipe.format_input()
    return testing(example)

def inflow_channel(dick):
    ex = pd.DataFrame(dick)
    print(ex.shape)
    pipe = DataPipeline(ex)
    example = pipe.format_input()
    return testing(example)


def plot_importances(model, X):
    """ Plots importance ranking of top features in Random Forest model
    """
    n = 10
    importances = model.feature_importances_[:n]
    std = np.std([tree.feature_importances_ for
                    tree in model.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]
    features = list(X.columns[indices])

    # Print the feature ranking
    print(f"\n{n}. Feature ranking:")

    for f in range(n):
        print("%d. %s (%f)" %
                (f + 1, features[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    _, ax = plt.subplots(figsize=(10, 15))

    ax.bar(range(n), importances[indices], yerr=std[indices],
            color="palevioletred", align="center")
    ax.set_xticks(range(n))
    ax.set_xticklabels(features, rotation=90)
    ax.set_xlim([-1, n])
    ax.set_xlabel("Importance")
    plt.xticks(rotation=30, ha='right')
    ax.set_title("Feature Importances")
    plt.show()


dick = {'approx_payout_date': 1363294800, 'body_length': 22457, 'channels': 0, 'country': 'US', 'currency': 'USD', 'delivery_method': 1.0, 'description': '<p style="text-indent: -13.241pt; margin-left: 13.241pt;"><span style="line-height: 119%; font-family: \'Bradley Hand ITC\'; font-size: 12pt; language: en-US; mso-default-font-family: \'Bradley Hand ITC\'; mso-ascii-font-family: \'Bradley Hand ITC\'; mso-latin-font-family: \'Bradley Hand ITC\'; mso-greek-font-family: Arial; mso-cyrillic-font-family: Arial; mso-armenian-font-family: TimesNewRomanPSMT; mso-hebrew-font-family: Arial; mso-arabic-font-family: Arial; mso-devanagari-font-family: TimesNewRomanPSMT; mso-bengali-font-family: TimesNewRomanPSMT; mso-gurmukhi-font-family: TimesNewRomanPSMT; mso-oriya-font-family: TimesNewRomanPSMT; mso-tamil-font-family: TimesNewRomanPSMT; mso-telugu-font-family: TimesNewRomanPSMT; mso-kannada-font-family: TimesNewRomanPSMT; mso-malayalam-font-family: TimesNewRomanPSMT; mso-thai-font-family: TimesNewRomanPSMT; mso-lao-font-family: TimesNewRomanPSMT; mso-tibetan-font-family: TimesNewRomanPSMT; mso-georgian-font-family: TimesNewRomanPSMT; mso-hangul-font-family: TimesNewRomanPSMT; mso-kana-font-family: TimesNewRomanPSMT; mso-bopomofo-font-family: TimesNewRomanPSMT; mso-han-font-family: TimesNewRomanPSMT; mso-halfwidthkana-font-family: TimesNewRomanPSMT; mso-yi-font-family: TimesNewRomanPSMT; mso-hansurrogate-font-family: TimesNewRomanPSMT; mso-nonhansurrogate-font-family: TimesNewRomanPSMT; mso-eudc-font-family: TimesNewRomanPSMT; mso-syriac-font-family: TimesNewRomanPSMT; mso-thaana-font-family: TimesNewRomanPSMT; mso-myanmar-font-family: TimesNewRomanPSMT; mso-sinhala-font-family: TimesNewRomanPSMT; mso-ethiopic-font-family: TimesNewRomanPSMT; mso-cherokee-font-family: TimesNewRomanPSMT; mso-canadianabor-font-family: TimesNewRomanPSMT; mso-ogham-font-family: TimesNewRomanPSMT; mso-runic-font-family: TimesNewRomanPSMT; mso-khmer-font-family: TimesNewRomanPSMT; mso-mongolian-font-family: TimesNewRomanPSMT; mso-braille-font-family: TimesNewRomanPSMT; mso-currency-font-family: TimesNewRomanPSMT; mso-asciisym-font-family: TimesNewRomanPSMT; mso-latinext-font-family: Arial; mso-ansi-language: en-US; mso-ligatures: none;" lang="en-US">&nbsp;</span><span style="font-size: large;"><strong><span style="line-height: 119%; font-family: \'Bradley Hand ITC\'; language: en-US; mso-default-font-family: \'Bradley Hand ITC\'; mso-ascii-font-family: \'Bradley Hand ITC\'; mso-latin-font-family: \'Bradley Hand ITC\'; mso-greek-font-family: Arial; mso-cyrillic-font-family: Arial; mso-armenian-font-family: TimesNewRomanPSMT; mso-hebrew-font-family: Arial; mso-arabic-font-family: Arial; mso-devanagari-font-family: TimesNewRomanPSMT; mso-bengali-font-family: TimesNewRomanPSMT; mso-gurmukhi-font-family: TimesNewRomanPSMT; mso-oriya-font-family: TimesNewRomanPSMT; mso-tamil-font-family: TimesNewRomanPSMT; mso-telugu-font-family: TimesNewRomanPSMT; mso-kannada-font-family: TimesNewRomanPSMT; mso-malayalam-font-family: TimesNewRomanPSMT; mso-thai-font-family: TimesNewRomanPSMT; mso-lao-font-family: TimesNewRomanPSMT; mso-tibetan-font-family: TimesNewRomanPSMT; mso-georgian-font-family: TimesNewRomanPSMT; mso-hangul-font-family: TimesNewRomanPSMT; mso-kana-font-family: TimesNewRomanPSMT; mso-bopomofo-font-family: TimesNewRomanPSMT; mso-han-font-family: TimesNewRomanPSMT; mso-halfwidthkana-font-family: TimesNewRomanPSMT; mso-yi-font-family: TimesNewRomanPSMT; mso-hansurrogate-font-family: TimesNewRomanPSMT; mso-nonhansurrogate-font-family: TimesNewRomanPSMT; mso-eudc-font-family: TimesNewRomanPSMT; mso-syriac-font-family: TimesNewRomanPSMT; mso-thaana-font-family: TimesNewRomanPSMT; mso-myanmar-font-family: TimesNewRomanPSMT; mso-sinhala-font-family: TimesNewRomanPSMT; mso-ethiopic-font-family: TimesNewRomanPSMT; mso-cherokee-font-family: TimesNewRomanPSMT; mso-canadianabor-font-family: TimesNewRomanPSMT; mso-ogham-font-family: TimesNewRomanPSMT; mso-runic-font-family: TimesNewRomanPSMT; mso-khmer-font-family: TimesNewRomanPSMT; mso-mongolian-font-family: TimesNewRomanPSMT; mso-braille-font-family: TimesNewRomanPSMT; mso-currency-font-family: TimesNewRomanPSMT; mso-asciisym-font-family: TimesNewRomanPSMT; mso-latinext-font-family: Arial; mso-ansi-language: en-US; mso-ligatures: none;" lang="en-US">SPEAKER:&nbsp; KIM GALGANO</span></strong></span></p>\r\n<p style="text-indent: -13.241pt; margin-left: 13.241pt;"><span style="line-height: 119%; font-family: \'Bradley Hand ITC\'; font-size: 12pt; language: en-US; mso-default-font-family: \'Bradley Hand ITC\'; mso-ascii-font-family: \'Bradley Hand ITC\'; mso-latin-font-family: \'Bradley Hand ITC\'; mso-greek-font-family: Arial; mso-cyrillic-font-family: Arial; mso-armenian-font-family: TimesNewRomanPSMT; mso-hebrew-font-family: Arial; mso-arabic-font-family: Arial; mso-devanagari-font-family: TimesNewRomanPSMT; mso-bengali-font-family: TimesNewRomanPSMT; mso-gurmukhi-font-family: TimesNewRomanPSMT; mso-oriya-font-family: TimesNewRomanPSMT; mso-tamil-font-family: TimesNewRomanPSMT; mso-telugu-font-family: TimesNewRomanPSMT; mso-kannada-font-family: TimesNewRomanPSMT; mso-malayalam-font-family: TimesNewRomanPSMT; mso-thai-font-family: TimesNewRomanPSMT; mso-lao-font-family: TimesNewRomanPSMT; mso-tibetan-font-family: TimesNewRomanPSMT; mso-georgian-font-family: TimesNewRomanPSMT; mso-hangul-font-family: TimesNewRomanPSMT; mso-kana-font-family: TimesNewRomanPSMT; mso-bopomofo-font-family: TimesNewRomanPSMT; mso-han-font-family: TimesNewRomanPSMT; mso-halfwidthkana-font-family: TimesNewRomanPSMT; mso-yi-font-family: TimesNewRomanPSMT; mso-hansurrogate-font-family: TimesNewRomanPSMT; mso-nonhansurrogate-font-family: TimesNewRomanPSMT; mso-eudc-font-family: TimesNewRomanPSMT; mso-syriac-font-family: TimesNewRomanPSMT; mso-thaana-font-family: TimesNewRomanPSMT; mso-myanmar-font-family: TimesNewRomanPSMT; mso-sinhala-font-family: TimesNewRomanPSMT; mso-ethiopic-font-family: TimesNewRomanPSMT; mso-cherokee-font-family: TimesNewRomanPSMT; mso-canadianabor-font-family: TimesNewRomanPSMT; mso-ogham-font-family: TimesNewRomanPSMT; mso-runic-font-family: TimesNewRomanPSMT; mso-khmer-font-family: TimesNewRomanPSMT; mso-mongolian-font-family: TimesNewRomanPSMT; mso-braille-font-family: TimesNewRomanPSMT; mso-currency-font-family: TimesNewRomanPSMT; mso-asciisym-font-family: TimesNewRomanPSMT; mso-latinext-font-family: Arial; mso-ansi-language: en-US; mso-ligatures: none;" lang="en-US">"I do love to speak, because it is, after all,&nbsp;the truths of God I get to share.&nbsp; It\'s with&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; authenticity and vulnerability that I share my struggles and failures,&nbsp;with the hope that some will catch the power&nbsp;Jesus can have in a life.&nbsp; </span></p>\r\n<p style="text-indent: -13.241pt; margin-left: 13.241pt;"><span style="line-height: 119%; font-family: \'Bradley Hand ITC\'; font-size: 12pt; language: en-US; mso-default-font-family: \'Bradley Hand ITC\'; mso-ascii-font-family: \'Bradley Hand ITC\'; mso-latin-font-family: \'Bradley Hand ITC\'; mso-greek-font-family: Arial; mso-cyrillic-font-family: Arial; mso-armenian-font-family: TimesNewRomanPSMT; mso-hebrew-font-family: Arial; mso-arabic-font-family: Arial; mso-devanagari-font-family: TimesNewRomanPSMT; mso-bengali-font-family: TimesNewRomanPSMT; mso-gurmukhi-font-family: TimesNewRomanPSMT; mso-oriya-font-family: TimesNewRomanPSMT; mso-tamil-font-family: TimesNewRomanPSMT; mso-telugu-font-family: TimesNewRomanPSMT; mso-kannada-font-family: TimesNewRomanPSMT; mso-malayalam-font-family: TimesNewRomanPSMT; mso-thai-font-family: TimesNewRomanPSMT; mso-lao-font-family: TimesNewRomanPSMT; mso-tibetan-font-family: TimesNewRomanPSMT; mso-georgian-font-family: TimesNewRomanPSMT; mso-hangul-font-family: TimesNewRomanPSMT; mso-kana-font-family: TimesNewRomanPSMT; mso-bopomofo-font-family: TimesNewRomanPSMT; mso-han-font-family: TimesNewRomanPSMT; mso-halfwidthkana-font-family: TimesNewRomanPSMT; mso-yi-font-family: TimesNewRomanPSMT; mso-hansurrogate-font-family: TimesNewRomanPSMT; mso-nonhansurrogate-font-family: TimesNewRomanPSMT; mso-eudc-font-family: TimesNewRomanPSMT; mso-syriac-font-family: TimesNewRomanPSMT; mso-thaana-font-family: TimesNewRomanPSMT; mso-myanmar-font-family: TimesNewRomanPSMT; mso-sinhala-font-family: TimesNewRomanPSMT; mso-ethiopic-font-family: TimesNewRomanPSMT; mso-cherokee-font-family: TimesNewRomanPSMT; mso-canadianabor-font-family: TimesNewRomanPSMT; mso-ogham-font-family: TimesNewRomanPSMT; mso-runic-font-family: TimesNewRomanPSMT; mso-khmer-font-family: TimesNewRomanPSMT; mso-mongolian-font-family: TimesNewRomanPSMT; mso-braille-font-family: TimesNewRomanPSMT; mso-currency-font-family: TimesNewRomanPSMT; mso-asciisym-font-family: TimesNewRomanPSMT; mso-latinext-font-family: Arial; mso-ansi-language: en-US; mso-ligatures: none;" lang="en-US">He is the only reason &ldquo;Hope is Here!&rdquo;</span></p>\r\n<p style="text-indent: -13.241pt; margin-left: 13.241pt;"><span style="line-height: 119%; font-family: \'Bradley Hand ITC\'; font-size: 12pt; language: en-US; mso-default-font-family: \'Bradley Hand ITC\'; mso-ascii-font-family: \'Bradley Hand ITC\'; mso-latin-font-family: \'Bradley Hand ITC\'; mso-greek-font-family: Arial; mso-cyrillic-font-family: Arial; mso-armenian-font-family: TimesNewRomanPSMT; mso-hebrew-font-family: Arial; mso-arabic-font-family: Arial; mso-devanagari-font-family: TimesNewRomanPSMT; mso-bengali-font-family: TimesNewRomanPSMT; mso-gurmukhi-font-family: TimesNewRomanPSMT; mso-oriya-font-family: TimesNewRomanPSMT; mso-tamil-font-family: TimesNewRomanPSMT; mso-telugu-font-family: TimesNewRomanPSMT; mso-kannada-font-family: TimesNewRomanPSMT; mso-malayalam-font-family: TimesNewRomanPSMT; mso-thai-font-family: TimesNewRomanPSMT; mso-lao-font-family: TimesNewRomanPSMT; mso-tibetan-font-family: TimesNewRomanPSMT; mso-georgian-font-family: TimesNewRomanPSMT; mso-hangul-font-family: TimesNewRomanPSMT; mso-kana-font-family: TimesNewRomanPSMT; mso-bopomofo-font-family: TimesNewRomanPSMT; mso-han-font-family: TimesNewRomanPSMT; mso-halfwidthkana-font-family: TimesNewRomanPSMT; mso-yi-font-family: TimesNewRomanPSMT; mso-hansurrogate-font-family: TimesNewRomanPSMT; mso-nonhansurrogate-font-family: TimesNewRomanPSMT; mso-eudc-font-family: TimesNewRomanPSMT; mso-syriac-font-family: TimesNewRomanPSMT; mso-thaana-font-family: TimesNewRomanPSMT; mso-myanmar-font-family: TimesNewRomanPSMT; mso-sinhala-font-family: TimesNewRomanPSMT; mso-ethiopic-font-family: TimesNewRomanPSMT; mso-cherokee-font-family: TimesNewRomanPSMT; mso-canadianabor-font-family: TimesNewRomanPSMT; mso-ogham-font-family: TimesNewRomanPSMT; mso-runic-font-family: TimesNewRomanPSMT; mso-khmer-font-family: TimesNewRomanPSMT; mso-mongolian-font-family: TimesNewRomanPSMT; mso-braille-font-family: TimesNewRomanPSMT; mso-currency-font-family: TimesNewRomanPSMT; mso-asciisym-font-family: TimesNewRomanPSMT; mso-latinext-font-family: Arial; mso-ansi-language: en-US; mso-ligatures: none;" lang="en-US">Hope is not wishful thinking. Hope is not something for which we strive and clamor for. Hope is not in our next vacation or&nbsp;&nbsp; visit to the pantry. Hope comes in unlikely places, like suffering. Hope is the anchor of our soul. Hope allows us to shine in a&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; desperate and hurting world.</span></p>\r\n<p style="text-indent: -13.241pt; margin-left: 13.241pt;"><span style="line-height: 119%; font-family: \'Bradley Hand ITC\'; font-size: 12pt; language: en-US; mso-default-font-family: \'Bradley Hand ITC\'; mso-ascii-font-family: \'Bradley Hand ITC\'; mso-latin-font-family: \'Bradley Hand ITC\'; mso-greek-font-family: Arial; mso-cyrillic-font-family: Arial; mso-armenian-font-family: TimesNewRomanPSMT; mso-hebrew-font-family: Arial; mso-arabic-font-family: Arial; mso-devanagari-font-family: TimesNewRomanPSMT; mso-bengali-font-family: TimesNewRomanPSMT; mso-gurmukhi-font-family: TimesNewRomanPSMT; mso-oriya-font-family: TimesNewRomanPSMT; mso-tamil-font-family: TimesNewRomanPSMT; mso-telugu-font-family: TimesNewRomanPSMT; mso-kannada-font-family: TimesNewRomanPSMT; mso-malayalam-font-family: TimesNewRomanPSMT; mso-thai-font-family: TimesNewRomanPSMT; mso-lao-font-family: TimesNewRomanPSMT; mso-tibetan-font-family: TimesNewRomanPSMT; mso-georgian-font-family: TimesNewRomanPSMT; mso-hangul-font-family: TimesNewRomanPSMT; mso-kana-font-family: TimesNewRomanPSMT; mso-bopomofo-font-family: TimesNewRomanPSMT; mso-han-font-family: TimesNewRomanPSMT; mso-halfwidthkana-font-family: TimesNewRomanPSMT; mso-yi-font-family: TimesNewRomanPSMT; mso-hansurrogate-font-family: TimesNewRomanPSMT; mso-nonhansurrogate-font-family: TimesNewRomanPSMT; mso-eudc-font-family: TimesNewRomanPSMT; mso-syriac-font-family: TimesNewRomanPSMT; mso-thaana-font-family: TimesNewRomanPSMT; mso-myanmar-font-family: TimesNewRomanPSMT; mso-sinhala-font-family: TimesNewRomanPSMT; mso-ethiopic-font-family: TimesNewRomanPSMT; mso-cherokee-font-family: TimesNewRomanPSMT; mso-canadianabor-font-family: TimesNewRomanPSMT; mso-ogham-font-family: TimesNewRomanPSMT; mso-runic-font-family: TimesNewRomanPSMT; mso-khmer-font-family: TimesNewRomanPSMT; mso-mongolian-font-family: TimesNewRomanPSMT; mso-braille-font-family: TimesNewRomanPSMT; mso-currency-font-family: TimesNewRomanPSMT; mso-asciisym-font-family: TimesNewRomanPSMT; mso-latinext-font-family: Arial; mso-ansi-language: en-US; mso-ligatures: none;" lang="en-US">Hope is Here!</span></p>\r\n<p style="text-indent: -13.241pt; margin-left: 13.241pt;"><span style="line-height: 119%; font-family: \'Bradley Hand ITC\'; font-size: 12pt; language: en-US; mso-default-font-family: \'Bradley Hand ITC\'; mso-ascii-font-family: \'Bradley Hand ITC\'; mso-latin-font-family: \'Bradley Hand ITC\'; mso-greek-font-family: Arial; mso-cyrillic-font-family: Arial; mso-armenian-font-family: TimesNewRomanPSMT; mso-hebrew-font-family: Arial; mso-arabic-font-family: Arial; mso-devanagari-font-family: TimesNewRomanPSMT; mso-bengali-font-family: TimesNewRomanPSMT; mso-gurmukhi-font-family: TimesNewRomanPSMT; mso-oriya-font-family: TimesNewRomanPSMT; mso-tamil-font-family: TimesNewRomanPSMT; mso-telugu-font-family: TimesNewRomanPSMT; mso-kannada-font-family: TimesNewRomanPSMT; mso-malayalam-font-family: TimesNewRomanPSMT; mso-thai-font-family: TimesNewRomanPSMT; mso-lao-font-family: TimesNewRomanPSMT; mso-tibetan-font-family: TimesNewRomanPSMT; mso-georgian-font-family: TimesNewRomanPSMT; mso-hangul-font-family: TimesNewRomanPSMT; mso-kana-font-family: TimesNewRomanPSMT; mso-bopomofo-font-family: TimesNewRomanPSMT; mso-han-font-family: TimesNewRomanPSMT; mso-halfwidthkana-font-family: TimesNewRomanPSMT; mso-yi-font-family: TimesNewRomanPSMT; mso-hansurrogate-font-family: TimesNewRomanPSMT; mso-nonhansurrogate-font-family: TimesNewRomanPSMT; mso-eudc-font-family: TimesNewRomanPSMT; mso-syriac-font-family: TimesNewRomanPSMT; mso-thaana-font-family: TimesNewRomanPSMT; mso-myanmar-font-family: TimesNewRomanPSMT; mso-sinhala-font-family: TimesNewRomanPSMT; mso-ethiopic-font-family: TimesNewRomanPSMT; mso-cherokee-font-family: TimesNewRomanPSMT; mso-canadianabor-font-family: TimesNewRomanPSMT; mso-ogham-font-family: TimesNewRomanPSMT; mso-runic-font-family: TimesNewRomanPSMT; mso-khmer-font-family: TimesNewRomanPSMT; mso-mongolian-font-family: TimesNewRomanPSMT; mso-braille-font-family: TimesNewRomanPSMT; mso-currency-font-family: TimesNewRomanPSMT; mso-asciisym-font-family: TimesNewRomanPSMT; mso-latinext-font-family: Arial; mso-ansi-language: en-US; mso-ligatures: none;" lang="en-US">I can&rsquo;t wait to get &ldquo;there&rdquo; and share in that hope with all of you! Until then&hellip;feel free&nbsp;&nbsp; to drop by and &ldquo;meet&rdquo; me at&nbsp;&nbsp;wwwchickswithchoices.com. &ldquo;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </span></p>\r\n<p style="text-indent: -13.241pt; margin-left: 13.241pt;"><span style="line-height: 119%; font-family: \'Bradley Hand ITC\'; font-size: 12pt; language: en-US; mso-default-font-family: \'Bradley Hand ITC\'; mso-ascii-font-family: \'Bradley Hand ITC\'; mso-latin-font-family: \'Bradley Hand ITC\'; mso-greek-font-family: Arial; mso-cyrillic-font-family: Arial; mso-armenian-font-family: TimesNewRomanPSMT; mso-hebrew-font-family: Arial; mso-arabic-font-family: Arial; mso-devanagari-font-family: TimesNewRomanPSMT; mso-bengali-font-family: TimesNewRomanPSMT; mso-gurmukhi-font-family: TimesNewRomanPSMT; mso-oriya-font-family: TimesNewRomanPSMT; mso-tamil-font-family: TimesNewRomanPSMT; mso-telugu-font-family: TimesNewRomanPSMT; mso-kannada-font-family: TimesNewRomanPSMT; mso-malayalam-font-family: TimesNewRomanPSMT; mso-thai-font-family: TimesNewRomanPSMT; mso-lao-font-family: TimesNewRomanPSMT; mso-tibetan-font-family: TimesNewRomanPSMT; mso-georgian-font-family: TimesNewRomanPSMT; mso-hangul-font-family: TimesNewRomanPSMT; mso-kana-font-family: TimesNewRomanPSMT; mso-bopomofo-font-family: TimesNewRomanPSMT; mso-han-font-family: TimesNewRomanPSMT; mso-halfwidthkana-font-family: TimesNewRomanPSMT; mso-yi-font-family: TimesNewRomanPSMT; mso-hansurrogate-font-family: TimesNewRomanPSMT; mso-nonhansurrogate-font-family: TimesNewRomanPSMT; mso-eudc-font-family: TimesNewRomanPSMT; mso-syriac-font-family: TimesNewRomanPSMT; mso-thaana-font-family: TimesNewRomanPSMT; mso-myanmar-font-family: TimesNewRomanPSMT; mso-sinhala-font-family: TimesNewRomanPSMT; mso-ethiopic-font-family: TimesNewRomanPSMT; mso-cherokee-font-family: TimesNewRomanPSMT; mso-canadianabor-font-family: TimesNewRomanPSMT; mso-ogham-font-family: TimesNewRomanPSMT; mso-runic-font-family: TimesNewRomanPSMT; mso-khmer-font-family: TimesNewRomanPSMT; mso-mongolian-font-family: TimesNewRomanPSMT; mso-braille-font-family: TimesNewRomanPSMT; mso-currency-font-family: TimesNewRomanPSMT; mso-asciisym-font-family: TimesNewRomanPSMT; mso-latinext-font-family: Arial; mso-ansi-language: en-US; mso-ligatures: none;" lang="en-US">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;~Kim </span></p>\r\n<p><span lang="en-US">&nbsp;</span></p>\r\n<p style="text-indent: -13.241pt; margin-bottom: 4pt; margin-left: 13.241pt;"><span style="font-size: large;"><strong><span style="color: black; line-height: 119%; font-family: \'Bradley Hand ITC\'; language: en-US; mso-default-font-family: \'Bradley Hand ITC\'; mso-ascii-font-family: \'Bradley Hand ITC\'; mso-latin-font-family: \'Bradley Hand ITC\'; mso-greek-font-family: Verdana; mso-cyrillic-font-family: Verdana; mso-armenian-font-family: Verdana; mso-hebrew-font-family: Verdana; mso-arabic-font-family: Arial; mso-thai-font-family: Verdana; mso-currency-font-family: Verdana; mso-latinext-font-family: Verdana; mso-ansi-language: en-US; mso-ligatures: none; mso-bidi-language: ar-SA;" lang="en-US">BAND:&nbsp; ETERNITY FOCUS</span></strong></span></p>\r\n<p style="text-indent: -13.241pt; margin-bottom: 4pt; margin-left: 13.241pt;"><span style="color: black; line-height: 119%; font-family: \'Bradley Hand ITC\'; font-size: 12pt; language: en-US; mso-default-font-family: \'Bradley Hand ITC\'; mso-ascii-font-family: \'Bradley Hand ITC\'; mso-latin-font-family: \'Bradley Hand ITC\'; mso-greek-font-family: Verdana; mso-cyrillic-font-family: Verdana; mso-armenian-font-family: Verdana; mso-hebrew-font-family: Verdana; mso-arabic-font-family: Arial; mso-thai-font-family: Verdana; mso-currency-font-family: Verdana; mso-latinext-font-family: Verdana; mso-ansi-language: en-US; mso-ligatures: none; mso-bidi-language: ar-SA;" lang="en-US">Eternity Focus is a group of contemporary Christian recording artists made up of four sisters; Alika, Danika, Janika and Lainika Seaman. Their </span><span style="line-height: 119%; font-family: \'Bradley Hand ITC\'; font-size: 12pt; language: en-US; mso-default-font-family: \'Bradley Hand ITC\'; mso-ascii-font-family: \'Bradley Hand ITC\'; mso-latin-font-family: \'Bradley Hand ITC\'; mso-greek-font-family: Verdana; mso-cyrillic-font-family: Verdana; mso-armenian-font-family: Verdana; mso-hebrew-font-family: Verdana; mso-arabic-font-family: Arial; mso-thai-font-family: Verdana; mso-currency-font-family: Verdana; mso-latinext-font-family: Verdana; mso-ansi-language: en-US; mso-ligatures: none;" lang="en-US">award-winning music has touched countless lives for Jesus Christ across the nation. They will share the inspiring&nbsp;&nbsp; message of God&rsquo;s grace, healing, hope and love through testimony and song. </span></p>\r\n<p style="text-indent: -13.241pt; margin-bottom: 4pt; margin-left: 13.241pt;"><span style="color: black; line-height: 119%; font-family: \'Bradley Hand ITC\'; font-size: 12pt; language: en-US; mso-default-font-family: \'Bradley Hand ITC\'; mso-ascii-font-family: \'Bradley Hand ITC\'; mso-latin-font-family: \'Bradley Hand ITC\'; mso-greek-font-family: Verdana; mso-cyrillic-font-family: Verdana; mso-armenian-font-family: Verdana; mso-hebrew-font-family: Verdana; mso-arabic-font-family: Arial; mso-thai-font-family: Verdana; mso-currency-font-family: Verdana; mso-latinext-font-family: Verdana; mso-ansi-language: en-US; mso-ligatures: none; mso-bidi-language: ar-SA;" lang="en-US"> With the release of their first self-titled recording project in April of 2007 their tour schedule has taken them to several states across the nation to participate in camps and retreats, youth and&nbsp;&nbsp; children&rsquo;s events, women&rsquo;s conferences, church services, Teens for Christ rallies, concerts and worship events. They were invited to compete in the National Fine Arts Festival for six years (2001-2007). Their fourth and newest album, Live the Lyrics, features original songs as well as fresh arrangements of some of today&rsquo;s top worship music. By listening to the message of hope found in Jesus, you will be inspired to truly live the lyrics.</span></p>\r\n<p><span lang="en-US">&nbsp;</span></p>', 'email_domain': 'westefc.org', 'event_created': 1361311394, 'event_end': 1362866400, 'event_published': 1361378496.0, 'event_start': 1362790800, 'fb_published': 0, 'gts': 2045.0, 'has_analytics': 0, 'has_header': None, 'has_logo': 1, 'listed': 'y', 'name': "HOPE IS HERE | Women's retreat", 'name_length': 30, 'num_order': 22, 'num_payouts': 5, 'object_id': 5563372, 'org_desc': '', 'org_facebook': 0.0, 'org_name': 'West Evangelical Free Church', 'org_twitter': 0.0, 'payee_name': 'West E. Free Church', 'payout_type': 'CHECK', 'previous_payouts': [{'address': '1161 N. Maize Rd.', 'amount': 250.42, 'country': 'US', 'created': '2012-01-19 03:11:44', 'event': 2574548, 'name': 'West E. Free Church', 'state': 'KS', 'uid': 24034936, 'zip_code': '67212'}, {'address': '1161 N. Maize Rd.', 'amount': 369.04, 'country': 'US', 'created': '2012-04-26 03:15:00', 'event': 3240215, 'name': 'West E. Free Church', 'state': 'KS', 'uid': 24034936, 'zip_code': '67212'}, {'address': '1161 N. Maize Rd.', 'amount': 39.54, 'country': 'US', 'created': '2012-07-19 03:14:35', 'event': 3796264, 'name': 'West E. Free Church', 'state': 'KS', 'uid': 24034936, 'zip_code': '67212'}, {'address': '1161 N. Maize Rd.', 'amount': 195.91, 'country': 'US', 'created': '2012-10-04 03:17:12', 'event': 4104790, 'name': 'West E. Free Church', 'state': 'KS', 'uid': 24034936, 'zip_code': '67212'}, {'address': '1161 N. Maize Rd.', 'amount': 79.08, 'country': 'US', 'created': '2012-10-25 03:16:30', 'event': 4399052, 'name': 'West E. Free Church', 'state': 'KS', 'uid': 24034936, 'zip_code': '67212'}, {'address': '1161 N. Maize Rd.', 'amount': 686.58, 'country': 'US', 'created': '2013-02-07 03:12:13', 'event': 5085292, 'name': 'West E. Free Church', 'state': 'KS', 'uid': 24034936, 'zip_code': '67212'}, {'address': '1161 N. Maize Rd.', 'amount': 1907.66, 'country': 'US', 'created': '2013-03-14 03:13:52', 'event': 5563372, 'name': 'West E. Free Church', 'state': 'KS', 'uid': 24034936, 'zip_code': '67212'}], 'sale_duration': 16.0, 'sale_duration2': 17, 'show_map': 1, 'ticket_types': [{'availability': 1, 'cost': 98.23, 'event_id': 5563372, 'quantity_sold': 2, 'quantity_total': 50}, {'availability': 1, 'cost': 88.78, 'event_id': 5563372, 'quantity_sold': 7, 'quantity_total': 50}, {'availability': 1, 'cost': 76.5, 'event_id': 5563372, 'quantity_sold': 0, 'quantity_total': 50}, {'availability': 1, 'cost': 69.88, 'event_id': 5563372, 'quantity_sold': 14, 'quantity_total': 50}, {'availability': 1, 'cost': 55.71, 'event_id': 5563372, 'quantity_sold': 2, 'quantity_total': 50}, {'availability': 1, 'cost': 0.0, 'event_id': 5563372, 'quantity_sold': 1, 'quantity_total': 10}], 'user_age': 447, 'user_created': 1322664283, 'user_type': 3, 'venue_address': '400 West Douglas Avenue', 'venue_country': 'US', 'venue_latitude': 37.686355, 'venue_longitude': -97.342702, 'venue_name': 'Drury Broadview', 'venue_state': 'KS'}
if __name__ == '__main__':
    # X, y = get_data('../data/data.json')
    # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=69)
   
    # test_script_examples(X)
    # X_test = get_example('../data/example.json')
    # X_test = test_inflow_channel('../data/data.json')
    X_test = inflow_channel(dick)
    # model = MyModel()
    # model.fit(X_train, y_train)
    # print(model.score(X_test, y_test))
    # y_pred = model.predict(X_test)
    # print(f1_score(y_test, y_pred))
   
    # f =  open('/Users/riley/Desktop/DSI/den-19/repos/Fraudy-McFraudson-2/src/model.pkl', 'wb')
    # pickle.dump(model, f)

    infile = open('../model.pkl','rb')
    model = pickle.load(infile)
    # plot_importances(model.model, X_train)
    # print(set(model.estimators_).difference(set(X_test.columns)))
    print(model.predict(X_test))

    # plot_importances(model, X_test)