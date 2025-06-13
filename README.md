# 🚀 Real-time Object Detection API

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-Ready-green.svg)](https://kubernetes.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-00a393.svg)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)

A **production-ready FastAPI service** for real-time object detection using YOLOv9 with comprehensive MLOps features including A/B testing, drift detection, active learning, and enterprise-grade monitoring.

## 🎯 What I Built & Why

This is a **complete MLOps solution** that transforms YOLOv9 object detection from a research model into an enterprise-ready service. I built this to demonstrate how to properly productionize machine learning models with all the necessary operational features that production systems require.

### **What's Included:**
- ✅ **Production FastAPI Service** with async processing and proper error handling
- ✅ **Optimized ONNX Runtime** for high-performance inference (<100ms)
- ✅ **A/B Testing Framework** for safe model deployments and experimentation
- ✅ **Real-time Drift Detection** using Evidently for data quality monitoring
- ✅ **Active Learning Pipeline** that automatically collects uncertain samples for retraining
- ✅ **Enterprise Monitoring** with Prometheus metrics and Grafana dashboards
- ✅ **Kubernetes-Native Deployment** with Helm charts and auto-scaling
- ✅ **Security-First Design** with non-root containers and proper RBAC
- ✅ **Complete CI/CD Pipeline** ready for GitOps workflows

## 🚀 Use Cases & Applications

### **🏭 Industrial & Manufacturing**
- **Quality Control**: Detect defects in products on assembly lines
- **Safety Monitoring**: Identify PPE compliance, unsafe behaviors, or hazardous conditions
- **Inventory Management**: Automated counting and tracking of parts/products
- **Predictive Maintenance**: Monitor equipment condition through visual inspection

### **🏪 Retail & E-commerce**
- **Inventory Management**: Real-time stock monitoring and automated replenishment
- **Loss Prevention**: Detect shoplifting, unusual behaviors, or security incidents
- **Customer Analytics**: People counting, demographic analysis, and behavior patterns
- **Product Recognition**: Automated checkout systems and inventory tracking

### **🚗 Transportation & Logistics**
- **Traffic Monitoring**: Vehicle counting, traffic flow analysis, and congestion detection
- **Fleet Management**: Vehicle tracking, driver behavior monitoring, and route optimization
- **Cargo Inspection**: Automated inspection of shipping containers and packages
- **Parking Management**: Occupancy detection and automated parking enforcement

### **🏥 Healthcare & Life Sciences**
- **Medical Imaging**: Automated analysis of X-rays, MRIs, and CT scans
- **Patient Monitoring**: Fall detection, patient activity tracking, and safety monitoring
- **Laboratory Automation**: Sample identification and quality control
- **Pharmaceutical**: Pill counting, packaging inspection, and compliance monitoring

### **🏢 Smart Buildings & Cities**
- **Security Systems**: Intrusion detection, perimeter monitoring, and access control
- **Crowd Management**: People counting, density monitoring, and flow optimization
- **Infrastructure Monitoring**: Structural inspection, maintenance needs assessment
- **Energy Management**: Occupancy-based HVAC and lighting optimization

### **🌾 Agriculture & Environment**
- **Crop Monitoring**: Disease detection, growth tracking, and yield estimation
- **Livestock Management**: Animal counting, health monitoring, and behavior analysis
- **Environmental Monitoring**: Wildlife tracking, pollution detection, and conservation
- **Precision Agriculture**: Automated harvesting and resource optimization

### **🎮 Media & Entertainment**
- **Content Moderation**: Automated detection of inappropriate content
- **Sports Analytics**: Player tracking, performance analysis, and game statistics
- **Augmented Reality**: Real-time object recognition for AR applications
- **Broadcasting**: Automated camera control and content enhancement

## 🎯 When to Use This Solution

### **✅ Perfect For:**
- **High-Volume Applications** requiring >1000 requests/hour
- **Production Environments** needing 99.9% uptime and reliability
- **Regulated Industries** requiring audit trails and compliance
- **A/B Testing Scenarios** where you need to compare model versions
- **Continuous Learning** systems that improve over time
- **Enterprise Deployments** needing monitoring and observability
- **Multi-Model Scenarios** where you want to run different versions

### **⚠️ Consider Alternatives If:**
- **Simple One-Off Tasks** where a script would suffice
- **Low-Volume Usage** (<100 requests/day) where serverless might be better
- **Prototype/Research** where model accuracy matters more than ops features
- **Constrained Resources** where you can't afford GPU infrastructure

## 🌟 Features

### 🎯 **Core Capabilities**
- **YOLOv9 Object Detection** with ONNX Runtime optimization
- **A/B Testing** via versioned endpoints (`/v1/predict` vs `/v2/predict`)
- **GPU Acceleration** with CUDA support
- **Real-time Inference** with <100ms latency
- **Scalable Architecture** supporting high-throughput workloads

### 🔬 **MLOps Features**
- **📊 Drift Detection**: Real-time data drift monitoring with Evidently
- **🧠 Active Learning**: Automatic uncertain sample collection for retraining
- **📈 Prometheus Metrics**: Comprehensive monitoring and alerting
- **🔄 Model Versioning**: Easy A/B testing and gradual rollouts
- **📋 Health Checks**: Kubernetes-ready liveness and readiness probes

### 🏗️ **Production Ready**
- **🐳 Optimized Docker**: Multi-stage builds <500MB
- **☸️ Kubernetes Native**: Complete Helm charts with auto-scaling
- **🔐 Security Hardened**: Non-root containers, security contexts
- **📊 Observability**: Prometheus + Grafana monitoring stack
- **🚀 High Availability**: Pod disruption budgets, anti-affinity rules

## 📋 Table of Contents

- [What I Built & Why](#-what-i-built--why)
- [Use Cases & Applications](#-use-cases--applications)
- [When to Use This Solution](#-when-to-use-this-solution)
- [Quick Start](#-quick-start)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Documentation](#-api-documentation)
- [Configuration](#-configuration)
- [Deployment](#-deployment)
- [Monitoring](#-monitoring)
- [Development](#-development)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

## ⚡ Quick Start

Get the API running in under 5 minutes:

```bash
# 1. Clone the repository
git clone <repository-url>
cd real-time-vision-api

# 2. Set up models and data
python setup_models.py --setup-all

# 3. Configure environment (optional)
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key

# 4. Start with Docker Compose
docker-compose up -d

# 5. Test the API
python test_api.py
```

**API Endpoints:**
- 🏥 Health: `http://localhost:8000/health`
- 📖 Docs: `http://localhost:8000/docs`
- 🔍 V1 Predict: `POST http://localhost:8000/v1/predict`
- 🔍 V2 Predict: `POST http://localhost:8000/v2/predict`
- 📊 Metrics: `http://localhost:8000/metrics`

## 🏗️ Architecture

![System Architecture](docs/images/architecture-diagram.png)

### **Technology Stack**

| Component | Technology | Purpose |
|-----------|------------|---------|
| **API Framework** | FastAPI 0.104.1 | High-performance async API |
| **ML Runtime** | ONNX Runtime 1.16.3 | Optimized inference engine |
| **Model** | YOLOv9 | State-of-the-art object detection |
| **Monitoring** | Prometheus + Grafana | Metrics and visualization |
| **Drift Detection** | Evidently | Data quality monitoring |
| **Storage** | S3 + Kubernetes PVC | Model and data persistence |
| **Orchestration** | Kubernetes + Helm | Container orchestration |
| **Security** | Non-root containers | Production security |

## 🛠️ Installation

### Prerequisites

- **Python 3.10+**
- **Docker & Docker Compose**
- **Kubernetes cluster** (for production)
- **NVIDIA GPU** (optional, for acceleration)
- **AWS S3 access** (for active learning)

### Local Development Setup

```bash
# Clone repository
git clone <repository-url>
cd real-time-vision-api

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up models and data
python setup_models.py --setup-all

# Configure environment variables
cp .env.example .env  # Edit with your settings
```

### Docker Setup

```bash
# Build the image
docker build -t object-detection-api:latest .

# Run with Docker Compose (includes monitoring)
docker-compose up -d

# Check container status
docker-compose ps
```

### Kubernetes Setup

```bash
# Install with Helm
helm upgrade --install object-detection-api ./helm-chart/object-detection-api \
  --namespace ml-services \
  --create-namespace \
  --set image.repository=your-registry/object-detection-api \
  --set image.tag=latest

# Check deployment
kubectl get pods -n ml-services
kubectl get services -n ml-services
```

## 🎯 Usage

### Basic Prediction

```python
import requests

# Upload image for detection
with open('image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/v1/predict',
        files={'file': f},
        data={'conf_threshold': 0.5}
    )

result = response.json()
print(f"Found {len(result['detections'])} objects")
```

### cURL Examples

```bash
# Test health endpoint
curl http://localhost:8000/health

# Object detection with v1 model
curl -X POST "http://localhost:8000/v1/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.jpg" \
  -F "conf_threshold=0.5"

# Object detection with v2 model
curl -X POST "http://localhost:8000/v2/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.jpg" \
  -F "conf_threshold=0.3"

# Check drift status
curl http://localhost:8000/drift/status

# Get Prometheus metrics
curl http://localhost:8000/metrics
```

### Response Format

```json
{
  "request_id": "uuid-string",
  "version": "v1",
  "detections": [
    {
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.95,
      "bbox": [100.0, 50.0, 200.0, 300.0]
    }
  ],
  "inference_time": 0.045,
  "image_shape": [480, 640, 3],
  "confidence_threshold": 0.5
}
```

## 📚 API Documentation

### Interactive Documentation

Once the service is running, access the interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check and model status |
| `POST` | `/v1/predict` | Object detection with model v1 |
| `POST` | `/v2/predict` | Object detection with model v2 |
| `GET` | `/metrics` | Prometheus metrics |
| `GET` | `/drift/status` | Data drift monitoring status |
| `POST` | `/reference/update` | Update drift reference data |

### Model Versions

- **v1**: Primary production model
- **v2**: A/B testing model or newer version

You can easily switch between models or implement traffic splitting for gradual rollouts.

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `WORKERS` | Number of Uvicorn workers | `1` |
| `LOG_LEVEL` | Logging level | `info` |
| `MODEL_V1_PATH` | Path to v1 ONNX model | `/app/models/yolov9_v1.onnx` |
| `MODEL_V2_PATH` | Path to v2 ONNX model | `/app/models/yolov9_v2.onnx` |
| `S3_BUCKET_NAME` | S3 bucket for active learning | `ml-active-learning` |
| `UNCERTAINTY_THRESHOLD` | Uncertainty threshold for active learning | `0.7` |
| `REFERENCE_DATA_PATH` | Path to drift reference data | `/app/data/reference_features.json` |
| `AWS_ACCESS_KEY_ID` | AWS access key | - |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key | - |
| `AWS_DEFAULT_REGION` | AWS region | `us-west-2` |

### Helm Configuration

Key values in `helm-chart/object-detection-api/values.yaml`:

```yaml
# Scaling
replicaCount: 2
autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10

# Resources
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
```

## 🚀 Deployment

### Production Checklist

- [ ] **Models**: Place real YOLOv9 ONNX models in `/models/`
- [ ] **GPU**: Ensure GPU nodes available in Kubernetes
- [ ] **Storage**: Configure persistent volumes for models/data
- [ ] **Monitoring**: Set up Prometheus and Grafana
- [ ] **Security**: Configure AWS IAM roles and secrets
- [ ] **Networking**: Set up ingress and load balancer
- [ ] **Scaling**: Configure HPA and resource limits

### Docker Deployment

```bash
# Production build
docker build -t your-registry/object-detection-api:v1.0.0 .
docker push your-registry/object-detection-api:v1.0.0

# Run in production mode
docker run -d \
  --name object-detection-api \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  -v $(pwd)/data:/app/data \
  -e WORKERS=4 \
  -e LOG_LEVEL=info \
  your-registry/object-detection-api:v1.0.0
```

### Kubernetes Deployment

```bash
# Deploy to production namespace
helm upgrade --install object-detection-api ./helm-chart/object-detection-api \
  --namespace production \
  --create-namespace \
  --set image.repository=your-registry/object-detection-api \
  --set image.tag=v1.0.0 \
  --set replicaCount=3 \
  --set resources.limits.memory=8Gi \
  --set autoscaling.enabled=true \
  --set monitoring.serviceMonitor.enabled=true \
  --set persistence.models.enabled=true \
  --set awsCredentials.accessKeyId=AKIA... \
  --set awsCredentials.secretAccessKey=secret...

# Verify deployment
kubectl get pods -n production
kubectl get services -n production
kubectl describe deployment object-detection-api -n production
```

### Traffic Splitting (A/B Testing)

Use Istio for advanced traffic management:

```yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: object-detection-split
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

## 📊 Monitoring

### Prometheus Metrics

The API exposes comprehensive metrics:

#### **Request Metrics**
- `object_detection_requests_total{version, status}` - Total requests
- `object_detection_request_duration_seconds{version}` - Request latency
- `object_detection_inference_duration_seconds{version}` - Inference time

#### **MLOps Metrics**
- `data_drift_detected_total{version}` - Drift detection events
- `active_learning_samples_total{version}` - Uncertain samples saved

#### **System Metrics**
- Standard FastAPI metrics (response times, error rates)
- Kubernetes metrics (CPU, memory, GPU utilization)

### Grafana Dashboards

Access monitoring dashboards:

- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090

Key dashboard panels:
- Request rate and latency trends
- Model performance comparison (v1 vs v2)
- Data drift alerts
- Resource utilization
- Error rate monitoring

### Alerting Rules

Example Prometheus alerting rules:

```yaml
groups:
- name: object-detection-api
  rules:
  - alert: HighLatency
    expr: histogram_quantile(0.95, object_detection_request_duration_seconds) > 1.0
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High API latency detected"
      
  - alert: DataDriftDetected
    expr: increase(data_drift_detected_total[1h]) > 0
    for: 0m
    labels:
      severity: warning
    annotations:
      summary: "Data drift detected in object detection model"
```

## 👩‍💻 Development

### Setting Up Development Environment

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest black isort mypy

# Run in development mode
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Run tests
python test_api.py

# Format code
black app.py
isort app.py

# Type checking
mypy app.py
```

### Model Development

```bash
# Convert PyTorch model to ONNX
python setup_models.py --convert model.pt --optimize

# Test model loading
python -c "
import onnxruntime as ort
session = ort.InferenceSession('models/yolov9_v1.onnx')
print('Model loaded successfully')
print(f'Input shape: {session.get_inputs()[0].shape}')
"
```

### Adding New Features

1. **New Endpoints**: Add to `app.py` with proper typing
2. **Metrics**: Use Prometheus client to add custom metrics
3. **Configuration**: Add environment variables and Helm values
4. **Tests**: Update `test_api.py` with new test cases

### Code Structure

```
real-time-vision-api/
├── app.py                          # Main FastAPI application
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Multi-stage container build
├── docker-compose.yml              # Local development stack
├── .dockerignore                   # Docker build optimization
├── setup_models.py                 # Model setup utilities
├── test_api.py                     # API testing suite
├── DEPLOYMENT.md                   # Deployment guide
├── helm-chart/                     # Kubernetes deployment
│   └── object-detection-api/
│       ├── Chart.yaml
│       ├── values.yaml
│       └── templates/
├── monitoring/                     # Monitoring configuration
│   ├── prometheus.yml
│   └── grafana/
└── models/                         # ONNX model files
    ├── yolov9_v1.onnx
    └── yolov9_v2.onnx
```

## 🔧 Troubleshooting

### Common Issues

#### **Model Loading Failed**
```bash
# Check model files
kubectl exec -it deployment/object-detection-api -- ls -la /app/models/

# Verify ONNX format
python -c "import onnxruntime; ort.InferenceSession('models/yolov9_v1.onnx')"
```

#### **GPU Not Available**
```bash
# Check GPU resources in cluster
kubectl describe nodes | grep -i gpu

# Verify NVIDIA device plugin
kubectl get pods -n kube-system | grep nvidia

# Test GPU in container
nvidia-smi
```

#### **High Memory Usage**
```bash
# Monitor resource usage
kubectl top pods -n ml-services

# Adjust memory limits
helm upgrade object-detection-api ./helm-chart/object-detection-api \
  --set resources.limits.memory=8Gi
```

#### **S3 Connection Issues**
```bash
# Test S3 connectivity
aws s3 ls s3://ml-active-learning/

# Check credentials
kubectl get secrets -n ml-services
kubectl describe secret object-detection-api-aws-credentials
```

### Performance Optimization

#### **Inference Optimization**
- Use GPU execution provider: `CUDA` > `CPU`
- Optimize batch size for your GPU memory
- Consider TensorRT for NVIDIA GPUs
- Use model quantization for faster inference

#### **Scaling Guidelines**
- **CPU-bound**: Increase worker count (`WORKERS` env var)
- **Memory-bound**: Scale vertically first (increase memory limits)
- **I/O-bound**: Scale horizontally (increase replica count)

#### **Container Optimization**
- Multi-stage builds reduce image size
- Layer caching optimizes build times
- Non-root user improves security
- Health checks enable proper orchestration

### Debugging

```bash
# View application logs
kubectl logs -f deployment/object-detection-api -n ml-services

# Debug container
kubectl exec -it deployment/object-detection-api -- /bin/bash

# Port forward for local access
kubectl port-forward service/object-detection-api 8000:80 -n ml-services

# Check resource usage
kubectl describe pod object-detection-api-xxx -n ml-services
```

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

### Development Process

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Code Standards

- **Python**: Follow PEP 8, use Black for formatting
- **Docker**: Multi-stage builds, minimal layers
- **Kubernetes**: Follow best practices, use labels
- **Documentation**: Update README and DEPLOYMENT.md

### Testing

```bash
# Run test suite
python test_api.py

# Test Docker build
docker build -t test-api .

# Test Helm chart
helm template ./helm-chart/object-detection-api
helm lint ./helm-chart/object-detection-api
```

## 📄 License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [YOLOv9](https://github.com/WongKinYiu/yolov9) - State-of-the-art object detection
- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework
- [ONNX Runtime](https://onnxruntime.ai/) - High-performance inference
- [Evidently](https://evidentlyai.com/) - ML monitoring and observability
- [Prometheus](https://prometheus.io/) - Systems monitoring toolkit

## 📞 Support

- **📖 Documentation**: Check [DEPLOYMENT.md](DEPLOYMENT.md) for detailed setup
- **🐛 Issues**: Open an issue on GitHub
- **💬 Discussions**: Use GitHub Discussions for questions
- **📧 Contact**: MLOps Team at mlops@yourcompany.com

---

<div align="center">

**⭐ Star this repo if it helped you!**

Made with ❤️ by Nguie Angoue jean roch junior "nguierochjunior@gmail.com"

</div>
