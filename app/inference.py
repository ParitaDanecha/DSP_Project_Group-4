from pathlib import Path

import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pickle
import numpy as np

from preprocessing import preprocessing
from training import train


ROOT_DIR = Path('../')
DATA_DIR = ROOT_DIR / 'data/house-prices'
MODELS_DIR = ROOT_DIR / 'models'

train_filepath = DATA_DIR / 'bigmart_Train-Set.csv'
test_filepath = DATA_DIR / 'bigmart_Test-Set.csv'

def inference(train_filepath, MODELS_DIR):
    df_processed, df_processed_target = preprocessing(train_filepath)
    model, X_test, y_test = train(df_processed, df_processed_target)

    # dumb the model using pickle
    # pickle_out = open(MODELS_DIR / 'model.pkl', "wb")
    # pickle.dump(model, pickle_out)
    # pickle_out.close()


    with open(MODELS_DIR/'model.pkl', 'wb') as f:
        pickle.dump(object, f)

    # predictions saved in dictionary
    predictions_dict = predictions(model, X_test, y_test)
    print("r2_score: ", predictions_dict["r2_score"], ",", "mean_absolute_error: ", predictions_dict["mean_absolute_error"], ",", "mean_squared_error: ", predictions_dict["mean_squared_error"])
    return predictions_dict

def predictions(model, X_test, y_test):
    y_pred = model.predict(X_test)
    predictions_dict = {
        "r2_score": r2_score(y_test, y_pred),
        "mean_absolute_error": mean_absolute_error(y_test, y_pred),
        "mean_squared_error": np.sqrt(mean_squared_error(y_test, y_pred))
    }
    return predictions_dict

inference(train_filepath, MODELS_DIR)
