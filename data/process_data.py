import sys
import pandas as pd
from sqlalchemy import create_engine

# Method to load data and concatenate. 
def load_data(messages_filepath, categories_filepath):
    '''
    Arguments:
        messages_filepath: File path of messages data
        categories_filepath: File path of categories data
    Output:
        df: Merged dataset from messages and categories
    '''

	# Reading messages and their categories usning pandas
	
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # Merge messages and categories to a single data frame based on 'id'
	
    df = pd.merge(messages, categories, on='id')
    
    return df

# Method to clean data and make data to be ready for analysis.
def clean_data(df):
    '''
    Arguments:
        df: Pandas dataframe with merged data
    Output:
        df: Cleaned dataset
    '''
	
    # Splitting categories based on semicolon
    categories = df['categories'].str.split(pat=';', expand=True)
    
    # Get the first row of the categories dataframe
    row = categories.iloc[0]
    
    # Lambda function to get ist of new column names for categories from the row.
	
    category_colnames = row.apply(lambda x: x[:-2])
    
    # Rename the categories columns
    categories.columns = category_colnames
    
    # Convert category values to just numbers 0 or 1
	
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
		
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    # Drop the original categories column from the df
    df.drop('categories', axis=1, inplace=True)
    
    # Concatenate the original dataframe with the new categories dataframe
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(subset='id', inplace=True)
    
    return df


# Method to save the cleaned data to the database.
def save_data(df, database_filename):
    '''    
    Inputs:
        df: cleaned dataset
        database_filename: database name
    Output: 
        A SQLite database
    '''
	
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterMessages', engine, index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()