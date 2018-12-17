# Disasters Response Project

## Motivation

This project is provided by Udacity and is in collaboration with FigureEight. In This project, a web app is created to read in a user message input and then assign up to 36 categories to this message input. In this classification process, I used random forest classifier combined with multipleoutput classifier to perform the task. The accuracy can reach up to 0.79. 


## What's in this repo?

* a demo of a Machine Learning python code - This is for my personal use before modulazing the code.
* a process_data python script. The input and output are database filepath and a cleaned data set ready to be fed into a ML pipeline
* a train_classifie python script. This script trains the MultipleOutputRegressor model so that it can classify new incoming messages.
* a run.py model which deploys the python script to an online app using a given database.

## ETL pipeline process

The process is standard.
* Used Sqlalchemy to connect with a target database and query data from it.
* After successful data retrieval, I wrote modulized functions to check missing data, tokenized text data.
* The end product of this pipeline is a Machine Learning ready dataset that contains Training and Test data.

## ML pipeline process

* Other than standartd ML process, I used grid search to tune some of the hyperparameters.
* __Note__: In order to save time, I only tuned two parameters. There are a lot more can be done to significantly improve the model perfomance.
* The end product of this ML process will produce 36 categories on any text message.
* This script saves the trained model in a pickle file.

## Instructions on using the web app

* Run the following commands in the project's root directory to set up your database and model.
* To run ETL pipeline that cleans data and stores in database
`python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

* To run ML pipeline that trains classifier and saves
`python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

* Run the following command in the app's directory to run your web app.
`python run.py`

* Go to http://0.0.0.0:3001/


__Acknowledgement__

1. All the errors are mine. This repo is for practicing purpose only.
2. The instructions on how to use the web app is directly taken from Udacity.

## Author
Renee Liu
