# Fraud Detection

No one likes being the victim of fraud and many companies do their best to protect their customers from fraudulent activity on their platforms. The burden of screening all transactions for this activity though is far too large for people. To that end we have devloped a model that flags incoming transactions as low, medium, and high risk allowing fraud officers to efficiently allocate their time.

## Data and Model Creation
Details on data featurization and model creation can be found in detail in model.md.

As a breif overview the training data and any incoming data is featurized with NLP methods and one-hot encoding of categorical features. The model used is a Random Forest Classifier and initial testing against a hold out set showed an accuracy of 98% and an F1 score of .88 (scores obtained with a default threshold of .5). The most important features of the model can be seen below.

Description of features in graph above.

## Web Interface

For ease of use, and to interface with the Heroku server which has incoming data, development of a Flask app hosted on AWS is currently underway.
