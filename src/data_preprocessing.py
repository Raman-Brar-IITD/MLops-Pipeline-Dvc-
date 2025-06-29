import os
import pandas as pd
import logging
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.preprocessing import LabelEncoder
nltk.download("stopwords")
nltk.download("punkt")
nltk.download('punkt_tab')

# Ensuring log dir exists
log_dir="logs"
os.makedirs(log_dir,exist_ok=True)

logger=logging.getLogger("data_preprocessing")
logger.setLevel("DEBUG")

console_handler=logging.StreamHandler()
console_handler.setLevel("DEBUG")

file_handler=logging.FileHandler(os.path.join(log_dir,"data_preprocessing.log"))
file_handler.setLevel('DEBUG')

formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def transform_text(text):
    """Transform input text by covertingit into lowercase,tokenizing, removing stopwards andpunctuations, and stemming"""
    ps=PorterStemmer()
    text.lower()
    text=nltk.word_tokenize(text)
    text=[word for word in text if word.isalnum()]
    text=[word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    text=[ps.stem(word) for word in text]
    return "".join(text)

def preprocess_df(df,text_column='text',target_column='target'):
    "Preprocesses the dataframe by encoding the target column,removing duplicates and transforming the text column."
    try:
        logger.debug("Starting encoding preprocessing for DataFrame")
        encoder=LabelEncoder()
        df[target_column]=encoder.fit_transform(df[target_column])
        logger.debug("Target column encoded")

        df=df.drop_duplicates(keep="first")
        logger.debug("duplicates removed")
        
        df.loc[:,text_column]=df[text_column].apply(transform_text)
        logger.debug("text column transformed")
        return df
    except KeyError as e:
        logger.error("column not found: %s ",e)
        raise
    except Exception as e:
        logger.error('Error during text encoding: %s', e)
        raise
def main(text_column="text",target_column='target'):
    """Main function to load raw data, preprocess it, and save the processed data."""
    try:
        train_data=pd.read_csv("./data/raw/train.csv")
        test_data=pd.read_csv("./data/raw/train.csv")

        train_processed=preprocess_df(train_data,text_column,target_column)
        test_processed=preprocess_df(test_data,text_column,target_column)

        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)
        
        train_processed.to_csv(os.path.join(data_path,"train_processed.csv"),index=False)
        test_processed.to_csv(os.path.join(data_path,"test_processed.csv"),index=False)
        logger.debug("Preprocessed data saved to: %s",data_path)
    except FileNotFoundError as e:
        logger.error("File not found : %s",e)
        raise
    except pd.errors.EmptyDataError as e:
        logger.error("No Data:%s",e)
        raise
    except Exception as e:
        logger.error("Failed to complete the data transformation process: %s", e)
        print(f"Error:{e}")

if __name__=='__main__':
    main()
    
