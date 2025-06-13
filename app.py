import asyncio
import io
import json
import logging
import os
import time
import uuid
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import boto3
import cv2
import numpy as np
import onnxruntime as ort
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.report import Report
from fastapi import FastAPI, File, HTTPException, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel
import pandas as pd
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrics
REQUEST_COUNT = Counter('object_detection_requests_total', 'Total requests', ['version', 'status'])
REQUEST_LATENCY = Histogram('object_detection_request_duration_seconds', 'Request latency', ['version'])
INFERENCE_LATENCY = Histogram('object_detection_inference_duration_seconds', 'Inference latency', ['version'])
DRIFT_DETECTED = Counter('data_drift_detected_total', 'Data drift detection events', ['version'])
ACTIVE_LEARNING_SAMPLES = Counter('active_learning_samples_total', 'Active learning samples saved', ['version'])

class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]

class PredictionResponse(BaseModel):
    request_id: str
    version: str
    detections: List[Detection]
    inference_time: float
    image_shape: Tuple[int, int, int]
    confidence_threshold: float

class ModelManager:
    def __init__(self):
        self.models = {}
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
        self.load_models()
        
    def load_models(self):
        """Load ONNX models for different versions"""
        model_paths = {
            'v1': os.getenv('MODEL_V1_PATH', 'models/yolov9_v1.onnx'),
            'v2': os.getenv('MODEL_V2_PATH', 'models/yolov9_v2.onnx')
        }
        
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        for version, path in model_paths.items():
            if os.path.exists(path):
                try:
                    session = ort.InferenceSession(path, providers=providers)
                    self.models[version] = session
                    logger.info(f"Loaded model {version} from {path}")
                except Exception as e:
                    logger.error(f"Failed to load model {version}: {e}")
            else:
                logger.warning(f"Model file not found: {path}")
    
    def preprocess_image(self, image: np.ndarray, input_size: int = 640) -> Tuple[np.ndarray, float]:
        """Preprocess image for YOLO inference"""
        height, width = image.shape[:2]
        scale = min(input_size / height, input_size / width)
        
        # Resize image
        new_height, new_width = int(height * scale), int(width * scale)
        resized_image = cv2.resize(image, (new_width, new_height))
        
        # Create padded image
        padded_image = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
        padded_image[:new_height, :new_width] = resized_image
        
        # Normalize and transpose
        input_tensor = padded_image.astype(np.float32) / 255.0
        input_tensor = np.transpose(input_tensor, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        return input_tensor, scale
    
    def postprocess_detections(self, outputs: np.ndarray, scale: float, 
                             conf_threshold: float = 0.5, iou_threshold: float = 0.4) -> List[Detection]:
        """Postprocess YOLO outputs to detections"""
        detections = []
        
        # outputs shape: [1, num_detections, 85] (4 bbox + 1 conf + 80 classes)
        predictions = outputs[0]
        
        for pred in predictions:
            # Extract box coordinates and confidence
            x_center, y_center, width, height = pred[:4]
            confidence = pred[4]
            
            if confidence < conf_threshold:
                continue
                
            # Get class scores and find best class
            class_scores = pred[5:]
            class_id = np.argmax(class_scores)
            class_confidence = class_scores[class_id] * confidence
            
            if class_confidence < conf_threshold:
                continue
            
            # Convert to absolute coordinates
            x1 = (x_center - width / 2) / scale
            y1 = (y_center - height / 2) / scale
            x2 = (x_center + width / 2) / scale
            y2 = (y_center + height / 2) / scale
            
            detection = Detection(
                class_id=int(class_id),
                class_name=self.class_names[class_id] if class_id < len(self.class_names) else "unknown",
                confidence=float(class_confidence),
                bbox=[float(x1), float(y1), float(x2), float(y2)]
            )
            detections.append(detection)
        
        # Apply NMS
        if detections:
            detections = self._apply_nms(detections, iou_threshold)
        
        return detections
    
    def _apply_nms(self, detections: List[Detection], iou_threshold: float) -> List[Detection]:
        """Apply Non-Maximum Suppression"""
        if not detections:
            return detections
        
        # Sort by confidence
        detections.sort(key=lambda x: x.confidence, reverse=True)
        
        keep = []
        while detections:
            current = detections.pop(0)
            keep.append(current)
            
            # Remove detections with high IoU
            detections = [det for det in detections 
                         if self._calculate_iou(current.bbox, det.bbox) < iou_threshold]
        
        return keep
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection over Union"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def predict(self, image: np.ndarray, version: str, conf_threshold: float = 0.5) -> List[Detection]:
        """Run inference on image"""
        if version not in self.models:
            raise ValueError(f"Model version {version} not available")
        
        session = self.models[version]
        input_tensor, scale = self.preprocess_image(image)
        
        # Run inference
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: input_tensor})
        
        return self.postprocess_detections(outputs[0], scale, conf_threshold)

class DriftDetector:
    def __init__(self):
        self.reference_data = None
        self.feature_buffer = []
        self.buffer_size = 100
        
    def extract_features(self, image: np.ndarray) -> Dict:
        """Extract features for drift detection"""
        # Basic image statistics
        features = {
            'mean_brightness': float(np.mean(image)),
            'std_brightness': float(np.std(image)),
            'mean_r': float(np.mean(image[:, :, 0])),
            'mean_g': float(np.mean(image[:, :, 1])),
            'mean_b': float(np.mean(image[:, :, 2])),
            'contrast': float(np.std(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))),
            'aspect_ratio': float(image.shape[1] / image.shape[0]),
            'resolution': float(image.shape[0] * image.shape[1])
        }
        return features
    
    def update_buffer(self, features: Dict):
        """Update feature buffer"""
        self.feature_buffer.append(features)
        if len(self.feature_buffer) > self.buffer_size:
            self.feature_buffer.pop(0)
    
    def detect_drift(self, version: str) -> bool:
        """Detect data drift using Evidently"""
        if len(self.feature_buffer) < 50 or self.reference_data is None:
            return False
        
        try:
            current_data = pd.DataFrame(self.feature_buffer[-50:])
            
            report = Report(metrics=[DataDriftPreset()])
            report.run(
                reference_data=self.reference_data,
                current_data=current_data
            )
            
            # Check if drift is detected
            result = report.as_dict()
            drift_detected = result['metrics'][0]['result']['dataset_drift']
            
            if drift_detected:
                DRIFT_DETECTED.labels(version=version).inc()
                logger.warning(f"Data drift detected for version {version}")
            
            return drift_detected
        except Exception as e:
            logger.error(f"Error in drift detection: {e}")
            return False
    
    def set_reference_data(self, features_list: List[Dict]):
        """Set reference data for drift detection"""
        self.reference_data = pd.DataFrame(features_list)

class ActiveLearningManager:
    def __init__(self):
        self.s3_client = None
        self.bucket_name = os.getenv('S3_BUCKET_NAME', 'ml-active-learning')
        self.uncertainty_threshold = float(os.getenv('UNCERTAINTY_THRESHOLD', '0.7'))
        
        # Initialize S3 client if credentials are available
        try:
            self.s3_client = boto3.client('s3')
            logger.info("S3 client initialized for active learning")
        except Exception as e:
            logger.warning(f"S3 client not initialized: {e}")
    
    def calculate_uncertainty(self, detections: List[Detection]) -> float:
        """Calculate prediction uncertainty"""
        if not detections:
            return 1.0  # High uncertainty for no detections
        
        confidences = [det.confidence for det in detections]
        
        # Use entropy-based uncertainty
        max_conf = max(confidences)
        uncertainty = 1.0 - max_conf
        
        return uncertainty
    
    async def save_uncertain_sample(self, image: np.ndarray, detections: List[Detection], 
                                  version: str, request_id: str):
        """Save uncertain samples to S3 for active learning"""
        uncertainty = self.calculate_uncertainty(detections)
        
        if uncertainty > self.uncertainty_threshold and self.s3_client:
            try:
                # Convert image to bytes
                _, buffer = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                image_bytes = buffer.tobytes()
                
                # Create metadata
                metadata = {
                    'request_id': request_id,
                    'version': version,
                    'uncertainty': str(uncertainty),
                    'num_detections': str(len(detections)),
                    'timestamp': str(time.time())
                }
                
                # Upload to S3
                key = f"uncertain_samples/{version}/{request_id}.jpg"
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=key,
                    Body=image_bytes,
                    Metadata=metadata
                )
                
                # Save detections metadata
                detection_data = {
                    'request_id': request_id,
                    'detections': [det.dict() for det in detections],
                    'uncertainty': uncertainty
                }
                
                metadata_key = f"uncertain_samples/{version}/{request_id}_metadata.json"
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=metadata_key,
                    Body=json.dumps(detection_data),
                    ContentType='application/json'
                )
                
                ACTIVE_LEARNING_SAMPLES.labels(version=version).inc()
                logger.info(f"Saved uncertain sample {request_id} to S3")
                
            except Exception as e:
                logger.error(f"Failed to save uncertain sample: {e}")

# Initialize components
app = FastAPI(
    title="Real-time Object Detection API",
    description="Production-ready YOLOv9 object detection with A/B testing and MLOps features",
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

# Initialize Prometheus instrumentator
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

# Initialize managers
model_manager = ModelManager()
drift_detector = DriftDetector()
active_learning_manager = ActiveLearningManager()

@app.on_event("startup")
async def startup_event():
    """Initialize reference data for drift detection"""
    logger.info("Starting Object Detection API")
    
    # Load reference data if available
    reference_path = os.getenv('REFERENCE_DATA_PATH', 'reference_features.json')
    if os.path.exists(reference_path):
        try:
            with open(reference_path, 'r') as f:
                reference_features = json.load(f)
                drift_detector.set_reference_data(reference_features)
                logger.info("Loaded reference data for drift detection")
        except Exception as e:
            logger.error(f"Failed to load reference data: {e}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "models": list(model_manager.models.keys())}

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()

async def process_detection(image_array: np.ndarray, version: str, 
                          conf_threshold: float, background_tasks: BackgroundTasks) -> PredictionResponse:
    """Process object detection with monitoring and active learning"""
    request_id = str(uuid.uuid4())
    
    try:
        # Start timing
        start_time = time.time()
        
        with REQUEST_LATENCY.labels(version=version).time():
            # Run inference
            inference_start = time.time()
            detections = model_manager.predict(image_array, version, conf_threshold)
            inference_time = time.time() - inference_start
            
            INFERENCE_LATENCY.labels(version=version).observe(inference_time)
        
        # Extract features for drift detection
        features = drift_detector.extract_features(image_array)
        drift_detector.update_buffer(features)
        
        # Check for drift
        background_tasks.add_task(drift_detector.detect_drift, version)
        
        # Active learning - save uncertain samples
        background_tasks.add_task(
            active_learning_manager.save_uncertain_sample,
            image_array, detections, version, request_id
        )
        
        REQUEST_COUNT.labels(version=version, status="success").inc()
        
        return PredictionResponse(
            request_id=request_id,
            version=version,
            detections=detections,
            inference_time=inference_time,
            image_shape=image_array.shape,
            confidence_threshold=conf_threshold
        )
        
    except Exception as e:
        REQUEST_COUNT.labels(version=version, status="error").inc()
        logger.error(f"Error in detection processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/predict", response_model=PredictionResponse)
async def predict_v1(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    conf_threshold: float = 0.5
):
    """Object detection endpoint - Version 1"""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        image_array = np.array(image)
        
        return await process_detection(image_array, "v1", conf_threshold, background_tasks)
        
    except Exception as e:
        logger.error(f"Error in v1 prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v2/predict", response_model=PredictionResponse)
async def predict_v2(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    conf_threshold: float = 0.5
):
    """Object detection endpoint - Version 2"""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        image_array = np.array(image)
        
        return await process_detection(image_array, "v2", conf_threshold, background_tasks)
        
    except Exception as e:
        logger.error(f"Error in v2 prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/drift/status")
async def get_drift_status():
    """Get current drift detection status"""
    return {
        "buffer_size": len(drift_detector.feature_buffer),
        "has_reference_data": drift_detector.reference_data is not None,
        "last_features": drift_detector.feature_buffer[-1] if drift_detector.feature_buffer else None
    }

@app.post("/reference/update")
async def update_reference_data():
    """Update reference data from current buffer"""
    if len(drift_detector.feature_buffer) >= 50:
        drift_detector.set_reference_data(drift_detector.feature_buffer[-50:])
        return {"message": "Reference data updated successfully"}
    else:
        raise HTTPException(status_code=400, detail="Not enough data in buffer")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)