import joblib
import pandas as pd

model = joblib.load("../models/loan_model.pkl")


def predict(data):
    df = pd.DataFrame[(data)]
    prediction = model.predict(df)
    return prediction