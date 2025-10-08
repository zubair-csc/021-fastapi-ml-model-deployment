# ML Model Deployment with FastAPI & Docker

A production-ready REST API for serving machine learning models using FastAPI and Docker.

## Features

- 🚀 FastAPI for high-performance API endpoints
- 🐳 Docker containerization for easy deployment
- 🔄 Hot model reloading without service restart
- 📊 Batch prediction support
- 🏥 Health check endpoints
- 🔒 Non-root user for security
- 📝 Automatic API documentation (Swagger/OpenAPI)
- 🌐 CORS support
- 🔧 Nginx reverse proxy configuration

## Project Structure

```
.
├── main.py              # FastAPI application
├── Dockerfile           # Docker container definition
├── docker-compose.yml   # Multi-container setup
├── requirements.txt     # Python dependencies
├── nginx.conf          # Nginx configuration
├── models/             # Directory for model files
│   └── model.pkl       # Your trained model
└── README.md           # This file
```

## Quick Start

### 1. Setup

```bash
# Clone or create the project directory
mkdir ml-api-deployment && cd ml-api-deployment

# Create models directory
mkdir models

# Place your trained model in the models directory
# cp /path/to/your/model.pkl models/model.pkl
```

### 2. Build and Run with Docker

```bash
# Build the Docker image
docker build -t ml-model-api .

# Run the container
docker run -d \
  --name ml-api \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  ml-model-api
```

### 3. Or use Docker Compose (Recommended)

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## API Endpoints

### Health Check
```bash
GET /health
```

### Model Information
```bash
GET /model/info
```

### Single Prediction
```bash
POST /predict
Content-Type: application/json

{
  "features": [5.1, 3.5, 1.4, 0.2]
}
```

### Batch Predictions
```bash
POST /predict/batch
Content-Type: application/json

{
  "instances": [
    [5.1, 3.5, 1.4, 0.2],
    [6.2, 2.9, 4.3, 1.3],
    [4.9, 3.0, 1.4, 0.2]
  ]
}
```

### Reload Model
```bash
POST /model/reload
```

## Testing the API

### Using cURL

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'

# Batch prediction
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"instances": [[5.1, 3.5, 1.4, 0.2], [6.2, 2.9, 4.3, 1.3]]}'
```

### Using Python

```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={"features": [5.1, 3.5, 1.4, 0.2]}
)
print(response.json())

# Batch prediction
response = requests.post(
    "http://localhost:8000/predict/batch",
    json={"instances": [
        [5.1, 3.5, 1.4, 0.2],
        [6.2, 2.9, 4.3, 1.3]
    ]}
)
print(response.json())
```

## Interactive Documentation

Access the automatic interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Training and Saving a Model

Example of training and saving a model:

```python
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Load dataset
X, y = load_iris(return_X_y=True)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model
joblib.dump(model, 'models/model.pkl')
```

## Environment Variables

- `MODEL_PATH`: Path to the model file (default: `/app/models/model.pkl`)
- `LOG_LEVEL`: Logging level (default: `info`)

## Production Deployment

### With Nginx Reverse Proxy

The docker-compose setup includes Nginx for:
- Load balancing
- SSL termination (add certificates)
- Request buffering
- Better performance

Access through Nginx: http://localhost:80

### Scaling

Scale the API service:

```bash
docker-compose up -d --scale ml-api=3
```

### Monitoring

Check container health:

```bash
docker ps
docker-compose ps
docker inspect ml-api
```

View logs:

```bash
docker-compose logs -f ml-api
```

## Security Considerations

- ✅ Non-root user in container
- ✅ Read-only model volume mount
- ✅ No sensitive data in logs
- ✅ Input validation with Pydantic
- ✅ CORS configuration
- 🔒 Consider adding API authentication
- 🔒 Use HTTPS in production
- 🔒 Implement rate limiting

## Troubleshooting

### Model not loading
```bash
# Check if model file exists
docker exec ml-api ls -la /app/models/

# Check logs
docker-compose logs ml-api
```

### Container won't start
```bash
# Check container logs
docker logs ml-api

# Rebuild without cache
docker-compose build --no-cache
```

### Port already in use
```bash
# Change port in docker-compose.yml
ports:
  - "8001:8000"
```

## Performance Optimization

1. **Use workers for parallel processing**:
   ```bash
   CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
   ```

2. **Enable model caching**

3. **Use async endpoints for I/O operations**

4. **Implement connection pooling for databases**

## License

MIT License - feel free to use this for your projects!

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.