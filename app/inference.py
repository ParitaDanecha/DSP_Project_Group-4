import pickle

import pandas as pd

from app.preprocessing import preprocessing

def inference(filepath, MODELS_DIR):
    df = pd.read_csv(filepath)
    if 'OutletSales' in df.columns:
        y = df.pop('OutletSales')
    # preprocessing file
    X = preprocessing(df)
    # model load
    with open(MODELS_DIR / 'model.pkl', 'rb') as pickle_file:
        model = pickle.load(pickle_file)
    arr = model.predict(X)
    return arr

def prediction_on_streamlitdata(formData : dict):
    df = pd.DataFrame([formData])
    #print(df)
    if 'OutletSales' in df.columns:
        y = df.pop('OutletSales')
    # preprocessing file
    X = preprocessing(df)
    # model load
    #with open(MODELS_DIR / 'model.pkl', 'rb') as pickle_file:
    with open('C:/Users/HP/DSP_Project_Group-4/models/model.pkl', 'rb') as pickle_file:
        model = pickle.load(pickle_file)
    arr = model.predict(X)
    
    arr = arr.tolist()
    return arr[0]