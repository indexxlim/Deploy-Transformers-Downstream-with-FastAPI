from fastapi import Depends, FastAPI
from pydantic import BaseModel
from model.model import Model, get_model


class PredReq(BaseModel):
    passage: str


class RredRep(BaseModel):
    prob: float


app = FastAPI()


@app.post("/generate", response_model=RredRep)
async def predict(request: PredReq, model: Model = Depends(get_model)):
    probabilities = model.predict(request.passage)
    return RredRep(prob=probabilities[1])
