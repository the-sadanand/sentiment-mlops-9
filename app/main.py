from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import os
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
app = FastAPI(title='Sentiment Analysis API')
model_name = os.getenv('MLFLOW_MODEL_NAME', 'Sentiment_model')
model_stage = os.getenv('MLFLOW_MODEL_STAGE', 'Production')

# Initialize model and preprocessor globally
sentiment_model = None
text_preprocessor = None

@app.on_event('startup')
async def load_model():
    global sentiment_model, text_preprocessor
    try:
        # Load latest production model and preprocessor from MLflow Registry
        # Ensure MLFLOW_TRACKING_URI is set in environment
        client = mlflow.tracking.MlflowClient()
        latest_version = client.get_latest_versions(model_name, stages=[model_stage])[0]
        model_uri = f'models:/{model_name}/{latest_version.version}'
        sentiment_model = mlflow.pyfunc.load_model(model_uri)
        
        # Assuming preprocessor was logged as a separate artifact with the model
        preprocessor_path = client.download_artifacts(run_id=latest_version.run_id, path='tfidf_vectorizer')
        # Load the preprocessor (e.g., using joblib for scikit-learn or custom loading)
        import joblib
        text_preprocessor = joblib.load(os.path.join(preprocessor_path, 'model.pkl')) # Adjust path if needed
        print(f'Loaded model {model_name} v{latest_version.version} from stage {model_stage}')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to load model: {str(e)}')

class TextInput(BaseModel):
    text: str

class PredictionOutput(BaseModel):
    sentiment: str
    confidence: float

@app.post('/predict', response_model=PredictionOutput)
async def predict_sentiment(input: TextInput):
    if not sentiment_model or not text_preprocessor:
        raise HTTPException(status_code=503, detail='Model not loaded')
    if not input.text.strip():
        raise HTTPException(status_code=400, detail='Input text cannot be empty')
    
    processed_text = text_preprocessor.transform([input.text]) # Apply preprocessor
    prediction = sentiment_model.predict(processed_text)[0]
    proba = sentiment_model.predict_proba(processed_text)[0]
    
    sentiment_label = 'positive' if prediction == 1 else 'negative' # Adjust based on your model output
    confidence_score = float(max(proba)) # Convert numpy float to Python float

    return PredictionOutput(sentiment=sentiment_label,
    confidence=confidence_score)
