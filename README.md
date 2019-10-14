# Disaster Response Pipeline Project

## Table of Contents
    
1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)
6. [Instructions](#instructions)

### Installation
We need Anaconda distribution of Python, and the following extra packages need to be installed for nltk:

* punkt
* wordnet
* stopwords

### Project Motivation

This project comprises of data analysis,engineering, NLP, and ML pipelines analyze message data that people sent during disasters to build a model for an API that classifies disaster messages. 

These messages can be used by appropriate disaster relief agencies.

### File Descriptions
There are three main foleders:

#### data
* disaster_categories.csv: dataset including all the categories of disaters.
* disaster_messages.csv: dataset including all the real time messages sent by people
* process_data.py: ETL pipeline scripts to read, clean, and save data into a database
* DisasterResponse.db: output of the ETL pipeline.

#### models
* train_classifier.py: machine learning pipeline scripts to train and export a classifier
* classifier.pkl: output of the machine learning pipeline, i.e. a trained classifer

#### app
* run.py: Python script to run the web application
* templates contains html file for the web applicatin

### Licensing, Authors, Acknowledgements
All credits must be given to Udacity for the inspiration and best quality study materials.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
