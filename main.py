from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import numpy as np
import joblib
import logging
from datetime import datetime
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ML Model Serving API",
    description="REST API for machine learning model predictions",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model = None
model_metadata = {}

# Pydantic models for request/response
class PredictionInput(BaseModel):
    features: List[float] = Field(..., description="Input features for prediction")
    
    class Config:
        json_schema_extra = {
            "example": {
                "features": [5.1, 3.5, 1.4, 0.2]
            }
        }

class PredictionOutput(BaseModel):
    prediction: Any
    probability: List[float] = None
    timestamp: str
    model_version: str

class BatchPredictionInput(BaseModel):
    instances: List[List[float]] = Field(..., description="Batch of input features")
    
    class Config:
        json_schema_extra = {
            "example": {
                "instances": [
                    [5.1, 3.5, 1.4, 0.2],
                    [6.2, 2.9, 4.3, 1.3]
                ]
            }
        }

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str
    version: str

def load_model():
    """Load the ML model from disk"""
    global model, model_metadata
    try:
        model_path = os.getenv("MODEL_PATH", "model.pkl")
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            model_metadata = {
                "version": "1.0.0",
                "loaded_at": datetime.now().isoformat(),
                "model_type": type(model).__name__
            }
            logger.info(f"Model loaded successfully: {model_metadata}")
        else:
            logger.warning(f"Model file not found at {model_path}")
            # Create a dummy model for demonstration
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.datasets import load_iris
            X, y = load_iris(return_X_y=True)
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit(X, y)
            model_metadata = {
                "version": "1.0.0-demo",
                "loaded_at": datetime.now().isoformat(),
                "model_type": "RandomForestClassifier (Demo)"
            }
            logger.info("Demo model created for testing")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("Starting up the application...")
    load_model()

@app.get("/", tags=["General"])
async def root():
    """Root endpoint"""
    return {
        "message": "ML Model Serving API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        timestamp=datetime.now().isoformat(),
        version=model_metadata.get("version", "unknown")
    )

@app.get("/model/info", tags=["Model"])
async def model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    return {
        "metadata": model_metadata,
        "model_params": model.get_params() if hasattr(model, 'get_params') else {}
    }

@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
async def predict(input_data: PredictionInput):
    """Make a single prediction"""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        # Reshape input for prediction
        features = np.array(input_data.features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Get prediction probabilities if available
        probabilities = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features)[0].tolist()
        
        return PredictionOutput(
            prediction=int(prediction) if isinstance(prediction, (np.integer, np.int64)) else float(prediction),
            probability=probabilities,
            timestamp=datetime.now().isoformat(),
            model_version=model_metadata.get("version", "unknown")
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(input_data: BatchPredictionInput):
    """Make batch predictions"""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        # Convert to numpy array
        features = np.array(input_data.instances)
        
        # Make predictions
        predictions = model.predict(features)
        
        # Get prediction probabilities if available
        probabilities = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features).tolist()
        
        return {
            "predictions": [int(p) if isinstance(p, (np.integer, np.int64)) else float(p) 
                          for p in predictions],
            "probabilities": probabilities,
            "count": len(predictions),
            "timestamp": datetime.now().isoformat(),
            "model_version": model_metadata.get("version", "unknown")
        }
    
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )

@app.post("/model/reload", tags=["Model"])
async def reload_model():
    """Reload the model from disk"""
    try:
        load_model()
        return {
            "status": "success",
            "message": "Model reloaded successfully",
            "metadata": model_metadata
        }
    except Exception as e:
        logger.error(f"Model reload error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model reload failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)