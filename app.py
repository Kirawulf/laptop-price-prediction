from fastapi import FastAPI, File, UploadFile
import pandas as pd, joblib
from io import BytesIO

app = FastAPI()
model = joblib.load("/content/drive/MyDrive/laptop_price_model.pkl")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    content = await file.read()
    df = pd.read_csv(BytesIO(content))
    preds = model.predict(df)
    return {"predictions": preds.tolist()}
