from pydantic import BaseModel

class TextInput(BaseModel):
    text: str

class PredictionOutput(BaseModel):
    sentiment: str
    confidence: float
