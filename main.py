from fastapi import FastAPI
import joblib
import numpy as np

model = joblib.load("modelo_random_forest.pkl")

app = FastAPI()

@app.get("/")
def home():
    return {"mensaje": "API de predicción de ventas"}

@app.post("/predict")
def predict(data: dict):

    features = np.array([[
        data["Price"],
        data["Quantity_Sold"],
        data["Age"]
    ]])

    prediction = model.predict(features)

    return {
        "Predicted_Total_Sales": float(prediction[0])
    }
