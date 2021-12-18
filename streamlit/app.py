import requests
import streamlit as st

st.title("Sales Price Predictor")

url = "http://127.0.0.1:8000/predict"

ProductID = st.text_input("Product ID", "FDW58")
Weight = st.text_input("Weight", 20.75)
FatContent =  st.selectbox('Fat Content',
        ('Low Fat', 'Regular'))
ProductVisibility = st.text_input("Product Visibility", 0.007564836)
ProductType =  st.selectbox('Product Type',
        ('Snack Foods', 'Dairy', 'Baking Goods', 'Breads', 'Breakfast', 'Canned',
         'Frozen Foods', 'Fruits and Vegetables', 'Hard Drinks', 'Household',
         'Meat', 'SeaFood', 'SnackFood', 'Soft Drinks', 'Startchy Foods'))
MRP = st.text_input("MRP", 107.8622)
OutletID = st.text_input("Outlet ID", "OUT049")
EstablishmentYear = st.text_input("Establishment Year", 1999)
OutletSize =  st.selectbox('Outlet Size',
        ('Medium', 'High', 'Small'))
LocationType =  st.selectbox('Location Type',
        ('Tier 1', 'Tier 2', 'Tier 3'))
OutletType = st.selectbox('Outlet Type',
        ('Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3', 'Grocery Store'))

myInput = {
    "ProductID" : ProductID,
    "Weight" : Weight,
    "FatContent" : FatContent,
    "ProductVisibility" : ProductVisibility,
    "ProductType" : ProductType,
    "MRP" : MRP,
    "OutletID" : OutletID,
    "EstablishmentYear" : EstablishmentYear,
    "OutletSize" : OutletSize,
    "LocationType" : LocationType,
    "OutletType" : OutletType
}

result=""
if st.button("Predict"):
    prediction = requests.post(url, json=myInput)
    st.success('The Prediction is: '+prediction.text)