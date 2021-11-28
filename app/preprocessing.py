import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

def preprocessing(filepath) -> pd.DataFrame:
    df_train = pd.read_csv(filepath)
    df_processed = missing_value_handling(df_train)
    df, df_target = feature_selection(df_processed)
    df_encoded = encoding_features(df)
    return df_encoded, df_target

# handling missing values
def missing_value_handling(df) -> pd.DataFrame:
    df['Weight'].fillna(df['Weight'].mean(), inplace=True)
    df['OutletSize'].fillna(df['OutletSize'].mode()[0], inplace=True)
    return df

# feature selection
def feature_selection(df) -> pd.DataFrame:
    df.drop(
        ['ProductID', 'OutletID', 'Weight', 'FatContent', 'ProductVisibility', 'EstablishmentYear', 'OutletSize',
         'LocationType', 'OutletType'], axis=1, inplace=True)
    df_target = df['OutletSales']
    df = df.drop(['OutletSales'], axis=1)
    return df, df_target

# One Hot Encoding for categorical data
def encoding_features(df) -> pd.DataFrame:
    transformer = ColumnTransformer(transformers=[('transform', OneHotEncoder(sparse=False), ["ProductType"])],
                                    remainder='passthrough')
    df = transformer.fit_transform(df)
    return df


