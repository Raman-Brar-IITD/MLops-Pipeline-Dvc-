import os
import pandas as pd
import pickle 
import logging
from sklearn.ensemble import RandomForestClassifier
import yaml
import numpy as np 

log_dir='logs'
os.makedirs(log_dir,exist_ok=True)

logger=logging.getLogger("Model Building")
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler=logging.FileHandler(os.path.join(log_dir,"model_building.log"))
file_handler.setLevel('DEBUG')

formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

def load_data(file_path:str)->pd.DataFrame:
    """Loads data from a CSV file
    :param file_path: Path to the CSV file
    :return: Loaded DataFrame"""
    
    try:
        df=pd.read_csv(file_path)
        logger.debug("Data loaded succesfully from :%s",file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error("Unable to Loade or Parse CSV Data :%s",e)
    
        raise
    except FileNotFoundError as e:
        logger.error("File not Found:%s",e)
        raise
    except Exception as e:
        logger.error("Unexpected error ocuured :%s ",e)
        raise

def train_model(X_train:np.ndarray,y_train:np.ndarray,params:dict)->RandomForestClassifier:
    """Train the RandomForest model.
    
    :param X_train: Training features
    :param y_train: Training labels
    :param params: Dictionary of hyperparameters
    :return: Trained RandomForestClassifier"""

    try:
        if X_train.shape[0] != y_train.shape[0]:
            logger.error("The number of samples in X_train and y_train are not same.")
            raise ValueError("The number of samples in X_train and y_train are not same.")

        logger.debug("Initilization Of Random Forest STARTED")
        clf=RandomForestClassifier(n_estimators=params['n_estimators'],random_state=params['random_state'])
        logger.debug('Model training started with %d samples', X_train.shape[0])
        clf.fit(X_train,y_train)
        logger.debug("Model training Completed")
        return clf
    except ValueError as e:
        logger.error('ValueError during model training: %s', e)
        raise
    except Exception as e:
        logger.error('Error during model training: %s', e)
        raise

def save_model(model,file_path:str)->None:
    """
    Save the trained model to a file.
    
    :param model: Trained model object
    :param file_path: Path to save the model file
    """
     
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        
        with open(file_path,'wb') as file:
            pickle.dump(model,file)
        logger.debug('Model saved to %s', file_path)
    except FileNotFoundError as e:
        logger.error('File path not found: %s', e)
        raise
    except Exception as e:
        logger.error('Error occurred while saving the model: %s', e)
        raise

def main():
    try:
        params = load_params('params.yaml')['model_building']
        train_data = load_data('./data/processed/train_tfidf.csv')
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values
        params = {'n_estimators': 100,'random_state': 3}
        clf = train_model(X_train, y_train, params) # type: ignore
        
        model_save_path = 'models/model.pkl'
        save_model(clf, model_save_path)

    except Exception as e:
        logger.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()


