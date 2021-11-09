# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 03:02:32 2021

@author: Parita Danecha
"""

import numpy as np
import pickle
import pandas as pd
import streamlit as st

from PIL import Image


pickle_in = open("rf.pkl","rb")
classifier=pickle.load(pickle_in)

def welcome():
    return "Welcome to the Big Mart Sales Prediction Application"

def predict_outlet_sales(ProductID, ProductType, OutletType, EstablishmentYear):
    prediction=classifier.predict([[ProductID, ProductType, OutletType, EstablishmentYear]])
    print(prediction)
    return prediction


def main():
    print("Inside main...")
    st.title("Outlet Sales Predictor")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Outlet Sales Predictor</h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    ProductID = st.text_input("ProductID","Type Here")
    ProductType = st.text_input("ProductType","Type Here")
    OutletType = st.text_input("OutletType","Type Here")
    EstablishmentYear = st.text_input("EstablishmentYear","Type Here")
    result=""
    if st.button("Predict"):
        result=predict_outlet_sales(ProductID, ProductType, OutletType, EstablishmentYear)
    st.success('The output is {}'.format(result))
    if st.button("About"):
        st.text("Built with Streamlit")

if __name__=='__main__':
    main()
