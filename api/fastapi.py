import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import psycopg2

from app.inference import prediction_on_streamlitdata

app = FastAPI()

# catch streamlit data
class request_body(BaseModel):
    ProductID : str
    Weight : float
    FatContent : str
    ProductVisibility : float
    ProductType : str
    MRP : float
    OutletID : str
    EstablishmentYear : int
    OutletSize : str
    LocationType : str
    OutletType : str

@app.get("/")
async def root():
    return {"message": "Welcome to Big Mart Sales Prediction"}

@app.post("/predict")
async def predict(stmlitdata: request_body):
    formData = stmlitdata.dict()
    prediction = prediction_on_streamlitdata(formData)

    OutletSales = prediction
    # save predictions to db
    connection = psycopg2.connect(user='postgres',
                                  password='123456',
                                  host='localhost',
                                  database='postgres')
    # Create a cursor
    cusrsor = connection.cursor()

    # Define the query
    insert_query = "INSERT INTO predictions VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"

    db_values = [formData['ProductID'], formData['Weight'], formData['FatContent'], formData['ProductVisibility'], formData['ProductType'], formData['MRP'], formData['OutletID'], formData['EstablishmentYear'], formData['OutletSize'], formData['LocationType'], formData['OutletType'], OutletSales]
    # Perform the query
    cusrsor.execute(insert_query, db_values)
    cusrsor.close()
    connection.commit()

    return prediction