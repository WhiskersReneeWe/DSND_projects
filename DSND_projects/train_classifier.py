import sys
# import libraries
import pandas as pd
from sqlalchemy import create_engine

#sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from  sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score, precision_score, recall_score
from statistics import mean
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import re
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

import pickle

def load_data(database_filepath):
    # load data from database
    database_filepath = 'sqlite:///' + database_filepath
    engine = create_engine(database_filepath)
    #print (engine.table_names()) -- debugging
    df = pd.read_sql_table('cleaned_disasters_msgs', engine)

    X = df['message']
    y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    cate_names =  y.columns.values
    return X, y, cate_names
    


def tokenize(text):
    tokens = [re.sub(r"[^a-zA-Z0-9\s]", "", stuff) for stuff in word_tokenize(text)]
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        if len(tok) < 1:  
            tokens.remove(tok)
        else:
            clean_tok = lemmatizer.lemmatize(tok).lower().strip()
            clean_tokens.append(clean_tok)
    
    return clean_tokens


def build_model():
    # Step1: build a ML pipeline
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer = tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputRegressor(RandomForestClassifier()))])
    
    # Step2: choose parameters for tuning 
#     parameters = {
#     #'clf__max_depth': [10, 15, 25, 30],
#     #'clf__n_estimators': [10, 20],
#     'vect__ngram_range': ((1, 1), (1, 2)),
#     'vect__max_df': (0.5, 0.75, 1.0),
#     'vect__max_features': (None, 5000, 10000),
#     'tfidf__use_idf': (True, False)
#                  }

    #cv = GridSearchCV(pipeline, param_grid=parameters)
    parameters = { 'clf__estimator__max_depth': [10, 15, 25, 30],
                   'clf__estimator__n_estimators': [10, 20], 
                   'vect__ngram_range': ((1, 1), (1, 2)), 
                   'vect__max_df': (0.5, 0.75, 1.0), 
                   'vect__max_features': (None, 5000, 10000), 
                   'tfidf__use_idf': (True, False) }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    #X,y = load_data(database_filepath)
    # model has been fited onto train datset
    y_pred = model.predict(X_test)
    accuracy = (y_pred == Y_test).mean()
    print("Labels:", category_names)
    print('Accuracy of tuned pipeline:', accuracy)

    
    for cate in range(Y_test.values.shape[1]):
        print(classification_report(Y_test.values.T[cate, :], y_pred.T[cate, :]))


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(DisasterResponse.db))
        X, Y, category_names = load_data(DisasterResponse.db)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        print('Building model...')
        model = build_model()
        # print(model)
        
        print('Training model...')
        model.fit(X_train, Y_train)
        #print('Am I here?')
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(DisasterResponse.db))
        save_model(model, DisasterResponse.db)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()