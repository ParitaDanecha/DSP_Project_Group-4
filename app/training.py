from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

def train(df_encoded, df_target):
    X_train, X_test, y_train, y_test = split_data(df_encoded, df_target)
    X_train_std, X_test_std = standardizing_features(X_train, X_test)
    model = model_fit(X_train_std, y_train)
    # model = model_fit(X_train, y_train)
    return model, X_test_std, y_test

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