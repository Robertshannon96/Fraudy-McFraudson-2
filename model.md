# Predictive Model Documentation

This is here to document the process of featurization, model creation, and tuning. The model at the heart of this application can be found in this directory as the model.pkl pickle file.

## Data Featurization

Training data and expected incoming data arrive in .json format and are moved in pandas dataframes. Of primary concern were features that were either categorical or plain text as we saw potential in them that we wanted to extract. Categorical features were one hot encoded, these included country, currency, and email domain. The plain text features were scrubbed first for any html with Beautiful Soup and then passed through tf-idf vectorizers from sklearn. These vectorizers were fit on our full training set and then pickled for future use with new data.

Additionally there were a few columns that contained binary information stored as strings, namely listed and account_type. These were changed over to binary encoding and account_type was pulled to function as our target variable for model training. Finally number of payouts and ticket types were pulled from their respective features into new features storing only the amount present.

## Model Training

A random forest classifier was used for the sake of simplicity and ease of tuning given our relatively short turn around on product. Initial training showed high accuracy with default parameters.

## Model Retraining

In the case of pickle file loss or further tuning of model the if name = main block in src/model.py has several lines of code that can be uncommented to run through model training and pickling of the result.