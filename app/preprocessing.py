import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import pickle
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder

ROOT_DIR = Path('../')
MODELS_DIR = ROOT_DIR / 'models'

def preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    df_processed = missing_value_handling(df)
    df = feature_selection(df_processed)
    df_encoded = encoding_features(df)
    return df_encoded

# handling missing values
def missing_value_handling(df) -> pd.DataFrame:
    df['Weight'].fillna(df['Weight'].mode()[0], inplace=True)
    df['OutletSize'].fillna(df['OutletSize'].mode()[0], inplace=True)
    return df

# feature selection
def feature_selection(df) -> pd.DataFrame:
    df.drop(
        ['ProductID', 'OutletID', 'Weight', 'FatContent', 'ProductVisibility', 'EstablishmentYear', 'OutletSize',
         'LocationType', 'OutletType'], axis=1, inplace=True)
    #df_target = df['OutletSales']
    if "OutletSales" in df.columns:
        df = df.drop(['OutletSales'], axis=1)
    #print(df)
    return df

# One Hot Encoding for categorical data
def encoding_features_old(df) -> pd.DataFrame:

    if Path(MODELS_DIR / 'encoder.pkl').exists():
        with open(MODELS_DIR / 'encoder.pkl', 'rb') as pickle_file:
            one_hot_encoder = pickle.load(pickle_file)
    else:
        with open(MODELS_DIR / 'encoder.pkl', 'wb') as f:
            one_hot_encoder = OneHotEncoder()
            pickle.dump(one_hot_encoder, f)

    one_hot_encoder.fit(df['ProductType'].values.reshape(-1, 1))
    encoded_categorical_data_matrix = one_hot_encoder.transform(df['ProductType'].values.reshape(-1, 1))
    encoded_data_columns = one_hot_encoder.get_feature_names(['ProductType'])
    encoded_categorical_data_df = pd.DataFrame.sparse.from_spmatrix(data=encoded_categorical_data_matrix,
                                                                    columns=encoded_data_columns, index=df.index)
    print(encoded_categorical_data_df)

    return encoded_categorical_data_df

# transformer = ColumnTransformer(transformers=[('transform', OneHotEncoder(sparse=False), ["ProductType"])],
#                                     remainder='passthrough')
#     df = transformer.fit_transform(df)
#     # df_col = df[['ProductType']]
#     # df_arr = transformer.fit_transform(df_col)
#     # df_enc = pd.DataFrame(df_arr, columns=['ProductType1', 'ProductType2',...])
#     # print(df)
#     return df

def encoding_features(df) -> pd.DataFrame:
    enc = LabelEncoder()
    df["ProductType"] = enc.fit_transform(df["ProductType"])
    return df