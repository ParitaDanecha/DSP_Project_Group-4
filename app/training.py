from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pickle
import pandas as pd

from preprocessing import preprocessing


def train(filepath, MODELS_DIR):
    df = pd.read_csv(filepath)
    y = df.pop('OutletSales')
    X = preprocessing(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    # X_train_std, X_test_std = standardizing_features(X_train, X_test)
    model = model_fit(X_train, y_train)

    with open(MODELS_DIR/'model.pkl', 'wb') as f:
        pickle.dump(model, f)

    # predictions saved in dictionary
    predictions_dict = predictions(model, X_test, y_test)
    print("r2_score: ", predictions_dict["r2_score"], ",", "mean_absolute_error: ",
          predictions_dict["mean_absolute_error"], ",", "mean_squared_error: ",
          predictions_dict["mean_squared_error"])
    return predictions_dict

# Split data
def split_data(df_encoded, df_target):
    X = df_encoded
    y = df_target
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=101, test_size=0.2)
    return X_train, X_test, y_train, y_test

# Standardizing train and test data
def standardizing_features(X_train, X_test):
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)
    return X_train_std, X_test_std

# Fit model
def model_fit(X_train_std, y_train):
    rf = RandomForestRegressor()
    rf.fit(X_train_std, y_train)
    return rf

def predictions(model, X_test, y_test):
    y_pred = model.predict(X_test)
    predictions_dict = {
        "r2_score": r2_score(y_test, y_pred),
        "mean_absolute_error": mean_absolute_error(y_test, y_pred),
        "mean_squared_error": np.sqrt(mean_squared_error(y_test, y_pred))
    }
    return predictions_dict