# Object Detection API Deployment Guide

This guide covers deployment of the production-ready YOLOv9 object detection API with MLOps features.

## 🏗️ Architecture Overview

The system includes:
- **FastAPI Application**: Real-time object detection with YOLOv9 ONNX models
- **A/B Testing**: Version endpoint routing (/v1/predict vs /v2/predict)
- **Monitoring**: Prometheus metrics + Evidently drift detection
- **Active Learning**: Uncertain sample storage to S3
- **Kubernetes**: Production deployment with Helm

## 📋 Prerequisites

### Required Infrastructure
- Kubernetes cluster with GPU nodes (NVIDIA Tesla V100/A100 recommended)
- NVIDIA Device Plugin for Kubernetes
- Prometheus + Grafana (for monitoring)
- S3-compatible storage (for active learning)
- Docker registry access

### Required Files
- YOLOv9 ONNX models (`yolov9_v1.onnx`, `yolov9_v2.onnx`)
- Reference feature data (`reference_features.json`)

## 🚀 Quick Start

### 1. Local Development

```bash
# Clone repository
git clone <repository-url>
cd real-time-vision-api

# Create model directory and add ONNX models
mkdir -p models data logs
# Place your YOLOv9 ONNX models in ./models/

# Set environment variables
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-west-2

# Run with Docker Compose
docker-compose up -d

# Test the API
curl -X POST "http://localhost:8000/v1/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_image.jpg" \
  -F "conf_threshold=0.5"
```

### 2. Build and Push Docker Image

```bash
# Build the image
docker build -t your-registry/object-detection-api:1.0.0 .

# Push to registry
docker push your-registry/object-detection-api:1.0.0
```

### 3. Kubernetes Deployment

```bash
# Install with Helm
cd helm-chart

# Update values for your environment
helm upgrade --install object-detection-api ./object-detection-api \
  --namespace ml-services \
  --create-namespace \
  --set image.repository=your-registry/object-detection-api \
  --set image.tag=1.0.0 \
  --set awsCredentials.accessKeyId=your_access_key \
  --set awsCredentials.secretAccessKey=your_secret_key \
  --set persistence.models.enabled=true \
  --set monitoring.serviceMonitor.enabled=true

# Check deployment
kubectl get pods -n ml-services
kubectl logs -f deployment/object-detection-api -n ml-services
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `WORKERS` | Number of Uvicorn workers | `1` |
| `LOG_LEVEL` | Logging level | `info` |
| `MODEL_V1_PATH` | Path to v1 ONNX model | `/app/models/yolov9_v1.onnx` |
| `MODEL_V2_PATH` | Path to v2 ONNX model | `/app/models/yolov9_v2.onnx` |
| `S3_BUCKET_NAME` | S3 bucket for active learning | `ml-active-learning` |
| `UNCERTAINTY_THRESHOLD` | Threshold for saving uncertain samples | `0.7` |
| `REFERENCE_DATA_PATH` | Path to reference features for drift detection | `/app/data/reference_features.json` |

### Helm Values

Key configuration options in `values.yaml`:

```yaml
# Scaling
replicaCount: 2
autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10

# Resources (adjust based on your GPU)
resources:
  limits:
    memory: 4Gi
    nvidia.com/gpu: 1
  requests:
    memory: 2Gi
    nvidia.com/gpu: 1

# Storage
persistence:
  models:
    enabled: true
    size: 10Gi
  data:
    enabled: true
    size: 5Gi
```

## 📊 Monitoring & Observability

### Prometheus Metrics

The API exposes these key metrics:

- `object_detection_requests_total` - Total requests by version and status
- `object_detection_request_duration_seconds` - Request latency by version
- `object_detection_inference_duration_seconds` - Inference latency by version
- `data_drift_detected_total` - Data drift detection events
- `active_learning_samples_total` - Active learning samples saved

### Health Checks

- **Liveness**: `GET /health` - Basic health check
- **Readiness**: `GET /health` - Ready to serve traffic
- **Metrics**: `GET /metrics` - Prometheus metrics

### Drift Detection

Monitor data drift status:

```bash
# Check drift detection status
curl http://localhost:8000/drift/status

# Update reference data
curl -X POST http://localhost:8000/reference/update
```

## 🧪 A/B Testing

The API supports model versioning through endpoint routing:

```bash
# Test Version 1
curl -X POST "http://localhost:8000/v1/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.jpg"

# Test Version 2
curl -X POST "http://localhost:8000/v2/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.jpg"
```

### Traffic Splitting with Istio

```yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: object-detection-traffic-split
spec:
  http:
  - match:
    - uri:
        prefix: "/predict"
  - route:
    - destination:
        host: object-detection-api
        subset: v1
      weight: 80
    - destination:
        host: object-detection-api
        subset: v2
      weight: 20
```

## 🔐 Security

### Production Security Checklist

- ✅ Non-root container user
- ✅ Read-only root filesystem (where possible)
- ✅ Dropped capabilities
- ✅ Security contexts configured
- ✅ Network policies (optional)
- ✅ Pod disruption budgets
- ✅ Service account with minimal permissions

### AWS IAM Permissions

For S3 active learning, ensure the service has these permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:PutObjectAcl"
      ],
      "Resource": "arn:aws:s3:::ml-active-learning/*"
    }
  ]
}
```

## 🚨 Troubleshooting

### Common Issues

**Model Loading Failed**
```bash
# Check if model files exist
kubectl exec -it deployment/object-detection-api -- ls -la /app/models/

# Check logs for model loading errors
kubectl logs deployment/object-detection-api | grep -i "model"
```

**GPU Not Available**
```bash
# Verify GPU resources
kubectl describe nodes | grep -i gpu

# Check NVIDIA device plugin
kubectl get pods -n kube-system | grep nvidia
```

**High Memory Usage**
```bash
# Monitor memory usage
kubectl top pods -n ml-services

# Adjust memory limits in values.yaml
helm upgrade object-detection-api ./object-detection-api \
  --set resources.limits.memory=8Gi
```

**S3 Connection Issues**
```bash
# Test S3 connectivity
kubectl exec -it deployment/object-detection-api -- \
  aws s3 ls s3://ml-active-learning/

# Check AWS credentials
kubectl get secrets -n ml-services
```

## 📈 Performance Optimization

### GPU Optimization
- Use CUDA execution provider for ONNX Runtime
- Optimize batch size for your GPU memory
- Consider TensorRT for NVIDIA GPUs

### Scaling Guidelines
- **CPU-bound**: Increase worker count
- **Memory-bound**: Scale vertically first
- **I/O-bound**: Scale horizontally

### Model Optimization
```bash
# Convert PyTorch to ONNX with optimization
python -c "
import torch
import onnx
from onnxruntime.tools import optimizer

# Load and optimize ONNX model
optimized_model = optimizer.optimize_model('yolov9.onnx')
optimized_model.save_model_to_file('yolov9_optimized.onnx')
"
```

## 🔄 CI/CD Pipeline

Example GitHub Actions workflow:

```yaml
name: Deploy Object Detection API
on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Build and push Docker image
      run: |
        docker build -t ${{ secrets.REGISTRY }}/object-detection-api:${{ github.sha }} .
        docker push ${{ secrets.REGISTRY }}/object-detection-api:${{ github.sha }}
    
    - name: Deploy to Kubernetes
      run: |
        helm upgrade --install object-detection-api ./helm-chart/object-detection-api \
          --namespace ml-services \
          --set image.tag=${{ github.sha }}
```

## 📚 API Documentation

Once deployed, access the interactive API documentation:

- **Swagger UI**: `http://your-domain/docs`
- **ReDoc**: `http://your-domain/redoc`

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review application logs: `kubectl logs deployment/object-detection-api`
3. Monitor metrics in Grafana/Prometheus
4. Open an issue in the repository

---

**Note**: This is a production-ready setup with enterprise-grade features including monitoring, scaling, security, and MLOps capabilities. Adjust configurations based on your specific requirements and infrastructure. 