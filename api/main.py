from fastapi import FastAPI
from pydantic import BaseModel
from model.predict import predict_parksay

app = FastAPI()


class InputText(BaseModel):
    text: str


@app.post("/predict")
def predict(input: InputText):
    is_parksay, confidence = predict_parksay(input.text)
    return {"is_parksay": bool(is_parksay), "confidence": confidence}
