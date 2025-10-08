# 🚀 FastAPI ML Model Deployment - Production REST API

## 📋 Project Overview
FastAPI ML Model Deployment is a comprehensive and production-ready REST API implementation for serving machine learning models. Built with clean architecture and modular design, it includes prediction endpoints, health monitoring, automatic API documentation, and model lifecycle management. The system supports both single and batch predictions with extensive customization options and comprehensive error handling.

## 🎯 Objectives
- Deploy machine learning models as scalable REST APIs with minimal setup
- Implement FastAPI best practices for high-performance model serving
- Support single and batch predictions with automatic input validation
- Provide real-time health monitoring and model information endpoints
- Enable hot model reloading without service restart for zero-downtime updates
- Offer comprehensive API documentation with interactive Swagger UI

## 📊 API Endpoints Information
| Endpoint | Method | Description |
|----------|--------|-------------|
| **/** | GET | Root endpoint with API information |
| **/health** | GET | Health check with model status |
| **/model/info** | GET | Detailed model metadata and parameters |
| **/predict** | POST | Single prediction with probability scores |
| **/predict/batch** | POST | Batch predictions for multiple inputs |
| **/model/reload** | POST | Hot reload model without restart |
| **/docs** | GET | Interactive Swagger API documentation |
| **/redoc** | GET | Alternative ReDoc documentation |

## 🔧 Technical Implementation

### 📌 API Architecture
- **Framework**: FastAPI with Pydantic validation for type safety
- **Model Loading**: Joblib-based persistence with automatic demo model fallback
- **Input Validation**: Pydantic BaseModel schemas with custom validators
- **Error Handling**: Comprehensive HTTP exception handling with detailed messages
- **Response Format**: JSON with prediction, probability, timestamp, and version info

### 🧹 Data Preprocessing
**Input Processing Pipeline:**
- Automatic numpy array conversion and reshaping
- Feature validation and dimension checking
- Missing value handling and error reporting
- Batch processing with efficient vectorization
- Type conversion for scikit-learn compatibility

### ⚙️ Model Serving Architecture
**Request Processing Flow:**
1. **Input Validation**: Pydantic schema validation
   - Type checking and constraint validation
   - Feature count verification
   - Data format normalization

2. **Prediction Pipeline**: Model inference
   - NumPy array preparation and reshaping
   - Model prediction with error catching
   - Probability calculation (if available)
   - Response formatting with metadata

3. **Response Generation**:
   - Prediction results with type conversion
   - Probability scores for classification
   - Timestamp and version information
   - Comprehensive error messages

### 📈 Monitoring Features
**Health and Observability:**
- Real-time health check endpoints
- Model loading status verification
- Detailed model metadata exposure
- Request/response logging
- Error tracking and reporting

**Model Management:**
- Hot reload functionality without downtime
- Model versioning and metadata tracking
- Graceful fallback to demo model
- Environment-based configuration

## 📊 Visualizations
- **Interactive API Docs**: Swagger UI with try-it-out functionality
- **API Schema**: OpenAPI 3.0 specification with detailed schemas
- **ReDoc Interface**: Beautiful alternative documentation view
- **Response Examples**: JSON response samples for all endpoints

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Required libraries: FastAPI, Uvicorn, scikit-learn, NumPy, Pydantic

### Installation
1. Clone the repository:
```bash
git clone https://github.com/zubair-csc/021-fastapi-ml-model-deployment.git
cd 021-fastapi-ml-model-deployment
```

2. Install required libraries:
```bash
pip install -r requirements.txt
```

### Model Setup
Place your trained model in the models directory:
```bash
mkdir models
# Copy your model file
cp /path/to/your/model.pkl models/model.pkl
```

Or train a sample model:
```python
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)
joblib.dump(model, 'models/model.pkl')
```

### Running the API
Execute the Python script:
```bash
python main.py
```

Or use Uvicorn directly:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

This will:
- Load the model from models directory
- Initialize FastAPI application
- Start Uvicorn server on port 8000
- Enable auto-reload for development
- Provide interactive documentation at /docs

## 📈 Usage Examples

### Health Check
```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2025-09-29T21:00:00",
  "version": "1.0.0"
}
```

### Single Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

Response:
```json
{
  "prediction": 0,
  "probability": [0.95, 0.03, 0.02],
  "timestamp": "2025-09-29T21:00:00",
  "model_version": "1.0.0"
}
```

### Batch Predictions
```python
import requests

response = requests.post(
    "http://localhost:8000/predict/batch",
    json={
        "instances": [
            [5.1, 3.5, 1.4, 0.2],
            [6.2, 2.9, 4.3, 1.3],
            [7.3, 2.9, 6.3, 1.8]
        ]
    }
)
print(response.json())
```

### Get Model Information
```bash
curl http://localhost:8000/model/info
```

### Reload Model
```bash
curl -X POST http://localhost:8000/model/reload
```

### Python Client Example
```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={"features": [5.1, 3.5, 1.4, 0.2]}
)
result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Probability: {result['probability']}")
```

## 🔮 Future Enhancements
- Authentication and API key management for secure access
- Rate limiting and request throttling for production use
- Prometheus metrics integration for monitoring
- Redis caching for frequently requested predictions
- Model A/B testing and canary deployments
- WebSocket support for real-time streaming predictions
- Model performance metrics and drift detection
- Integration with MLflow for model registry

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 🙌 Acknowledgments
- **FastAPI** for the high-performance web framework
- **Pydantic** for data validation and settings management
- **Uvicorn** for ASGI server implementation
- **scikit-learn** for machine learning models
- Open source community for continuous support and inspiration

## 📞 Contact
Zubair - [GitHub Profile](https://github.com/zubair-csc)

Project Link: [https://github.com/zubair-csc/021-fastapi-ml-model-deployment](https://github.com/zubair-csc/021-fastapi-ml-model-deployment)

⭐ Star this repository if you found it helpful!
