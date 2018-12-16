# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 10:25:12 2018
ML pipeline
@author: shiyan
"""

import nltk
nltk.download(['punkt', 'wordnet'])

# import libraries
import pandas as pd
from sqlalchemy import create_engine

#sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
#from sklearn.metrics import classification_report
from io import StringIO
from  sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score, precision_score, recall_score
from statistics import mean
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import re
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

import pickle

# load data from database
engine = create_engine('sqlite:///InsertDatabaseName.db')
#print (engine.table_names())
df = pd.read_sql_table('InsertTableName', engine)

X = df['message']
y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)

def tokenize(text): 
    tokens = [re.sub(r"[^a-zA-Z0-9\s]", "", stuff) for stuff in word_tokenize(text)]
#     for item in tokens:
#         if len(item) < 1:  
#             tokens.remove(item)
        
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        if len(tok) < 1:  
            tokens.remove(tok)
        else:
            clean_tok = lemmatizer.lemmatize(tok).lower().strip()
            clean_tokens.append(clean_tok)
    
    return clean_tokens

#ML pipeline builder
pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer = tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputRegressor(RandomForestClassifier()))])

# Hyperparameters that can be tuned for
MultiOutputRegressor(RandomForestClassifier()).get_params().keys()

# perform train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=109)
# fit the pipeline




for cate in range(y_test.values.shape[1]):
    print(classification_report(y_test.values.T[cate, :], y_pred.T[cate, :]))

parameters = {
    'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
    'features__text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
    'features__text_pipeline__vect__max_features': (None, 5000, 10000),
    'features__text_pipeline__tfidf__use_idf': (True, False)
}

pipeline = GridSearchCV(pipeline, param_grid=parameters)

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

accuracy = (y_pred == y_test).mean()
print('Accuracy of tuned pipeline:', accuracy)
for cate in range(y_test.values.shape[1]):
    print(classification_report(y_test.values.T[cate, :], y_pred.T[cate, :]))
    
filename = 'trained_ml_pipeline.sav'
pickle.dump(pipeline, open(filename, 'wb'))

