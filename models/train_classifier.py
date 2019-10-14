import sys
import pandas as pd
from sqlalchemy import create_engine
import nltk
import re
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV

nltk.download('stopwords')


#Method for loading data from db and extract data.

def load_data(database_filepath):
    '''
    Load data from database as dataframe
    Arguments:
        database_filepath: File path of sql database
    Output:
        X: Messages
        Y: Message categories
        category_names: Labels for All categories
    '''
	
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterMessages', engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    category_names = list(df.columns[4:])

    return X, Y, category_names



#Method for clean the data and tokenize.

def tokenize(text):
    '''
    Input:
        text: original message text
    Output:
        Cleaned text
    '''
    # Normalize Text to lowercase.
    text = re.sub(r"[^a-zA-Z0-9]", ' ', text.lower())
    words = word_tokenize(text)
    words = [w for w in words if w not in stopwords.words('english')]

    lemmatizer = WordNetLemmatizer()
    lemmed_text = [lemmatizer.lemmatize(w, pos='n').strip() for w in words]
    lemmed_text = [lemmatizer.lemmatize(w, pos='v').strip() for w in lemmed_text]
    
    return lemmed_text


#Method for building a ML pipeline to model the data.   
 
def build_model():
    
    pipeline = Pipeline([
                        ('vect', CountVectorizer(tokenizer=tokenize)),
                        ('tfidf', TfidfTransformer()),
                        ('clf', MultiOutputClassifier(RandomForestClassifier()))
                        ])

    parameters = {'clf__estimator__n_estimators': [50, 100],
                  'clf__estimator__min_samples_split': [2, 3, 4],
                  'clf__estimator__criterion': ['entropy', 'gini']
                 }
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


#Method to perorm evaluation metrics on test data

def evaluate_model(model, X_test, Y_test, category_names):
    '''    
    Input: 
        model: Model to be evaluated
        X_test: Test data (features)
        Y_test: True lables for Test data
        category_names: Labels for 36 categories
    Output:
        Print accuracy and classfication report for each category
    '''
    Y_pred = model.predict(X_test)

    for i in range(len(category_names)):
        print("Category:", category_names[i],"\n", classification_report(Y_test.iloc[:, i].values, Y_pred[:, i]))
        print('Accuracy of %25s: %.2f' %(category_names[i], accuracy_score(Y_test.iloc[:, i].values, Y_pred[:,i])))


# Method for saving the model as a pickle file.

def save_model(model, path):
    '''
    Input: 
        model: Model to be saved
        path: path of the output pickle file
    Output:
        A pickle file of saved model
    '''
    pickle.dump(model, open(path, "wb"))
    

#Main method for executing the ML pipeline.
def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        
        from workspace_utils import active_session
 
        with active_session():
            model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()