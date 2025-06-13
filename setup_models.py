#!/usr/bin/env python3
"""
Model setup script for YOLOv9 Object Detection API

This script helps download and convert YOLOv9 models to ONNX format.
"""

import os
import sys
import urllib.request
import hashlib
from pathlib import Path
import json

def download_file(url: str, destination: str, expected_hash: str = None):
    """Download a file with progress bar and hash verification"""
    print(f"📥 Downloading {os.path.basename(destination)}...")
    
    def progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(f"\r  Progress: {percent}% [{count * block_size}/{total_size} bytes]")
        sys.stdout.flush()
    
    urllib.request.urlretrieve(url, destination, progress_hook)
    print()  # New line after progress
    
    if expected_hash:
        print("🔍 Verifying file integrity...")
        with open(destination, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        
        if file_hash == expected_hash:
            print("✅ File integrity verified")
        else:
            print(f"❌ Hash mismatch! Expected: {expected_hash}, Got: {file_hash}")
            return False
    
    return True

def convert_pytorch_to_onnx(model_path: str, output_path: str, input_size: int = 640):
    """Convert PyTorch model to ONNX format"""
    try:
        import torch
        print(f"🔄 Converting {model_path} to ONNX...")
        
        # Load the model
        model = torch.load(model_path, map_location='cpu')
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, input_size, input_size)
        
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        print(f"✅ ONNX model saved to {output_path}")
        return True
        
    except ImportError:
        print("❌ PyTorch not available. Cannot convert model.")
        return False
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return False

def optimize_onnx_model(input_path: str, output_path: str):
    """Optimize ONNX model for inference"""
    try:
        from onnxruntime.tools import optimizer
        print(f"⚡ Optimizing ONNX model...")
        
        # Optimize the model
        optimized_model = optimizer.optimize_model(input_path)
        optimized_model.save_model_to_file(output_path)
        
        print(f"✅ Optimized model saved to {output_path}")
        return True
        
    except ImportError:
        print("❌ ONNX Runtime tools not available. Skipping optimization.")
        return False
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        return False

def create_sample_reference_data():
    """Create sample reference data for drift detection"""
    print("📊 Creating sample reference data...")
    
    # Sample feature data (replace with real data in production)
    reference_features = []
    
    import random
    for i in range(100):
        features = {
            'mean_brightness': random.uniform(100, 150),
            'std_brightness': random.uniform(30, 50),
            'mean_r': random.uniform(120, 140),
            'mean_g': random.uniform(120, 140),
            'mean_b': random.uniform(120, 140),
            'contrast': random.uniform(40, 60),
            'aspect_ratio': random.uniform(1.2, 1.8),
            'resolution': random.uniform(300000, 500000)
        }
        reference_features.append(features)
    
    # Save to file
    os.makedirs('data', exist_ok=True)
    with open('data/reference_features.json', 'w') as f:
        json.dump(reference_features, f, indent=2)
    
    print("✅ Sample reference data created at data/reference_features.json")
    return True

def download_yolov9_models():
    """Download YOLOv9 models"""
    print("🚀 Setting up YOLOv9 models...")
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Note: These are example URLs. In practice, you would:
    # 1. Train your own YOLOv9 models
    # 2. Convert them to ONNX
    # 3. Host them on your infrastructure
    
    print("⚠️ Note: You need to provide your own YOLOv9 ONNX models.")
    print("This script creates placeholder files. Replace them with real models.")
    
    # Create placeholder model files
    model_paths = [
        "models/yolov9_v1.onnx",
        "models/yolov9_v2.onnx"
    ]
    
    for model_path in model_paths:
        if not os.path.exists(model_path):
            print(f"📝 Creating placeholder: {model_path}")
            with open(model_path, 'w') as f:
                f.write("# Placeholder ONNX model file\n")
                f.write("# Replace this with your actual YOLOv9 ONNX model\n")
    
    print("\n📋 Model Setup Instructions:")
    print("1. Train YOLOv9 models using the official repository:")
    print("   https://github.com/WongKinYiu/yolov9")
    print("2. Convert PyTorch models to ONNX using this script:")
    print("   python setup_models.py --convert model.pt")
    print("3. Place ONNX files in the models/ directory")
    print("4. Update model paths in environment variables if needed")
    
    return True

def main():
    """Main setup function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup YOLOv9 models for Object Detection API")
    parser.add_argument("--convert", help="Convert PyTorch model to ONNX", metavar="MODEL_PATH")
    parser.add_argument("--optimize", help="Optimize ONNX model", metavar="MODEL_PATH")
    parser.add_argument("--setup-all", action="store_true", help="Setup everything (models + reference data)")
    parser.add_argument("--input-size", type=int, default=640, help="Input size for ONNX conversion")
    
    args = parser.parse_args()
    
    print("🔧 YOLOv9 Object Detection API - Model Setup")
    print("=" * 50)
    
    if args.convert:
        model_path = args.convert
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            return 1
        
        output_path = model_path.replace('.pt', '.onnx')
        success = convert_pytorch_to_onnx(model_path, output_path, args.input_size)
        
        if success and args.optimize:
            optimized_path = output_path.replace('.onnx', '_optimized.onnx')
            optimize_onnx_model(output_path, optimized_path)
        
        return 0 if success else 1
    
    elif args.optimize:
        model_path = args.optimize
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            return 1
        
        optimized_path = model_path.replace('.onnx', '_optimized.onnx')
        success = optimize_onnx_model(model_path, optimized_path)
        return 0 if success else 1
    
    elif args.setup_all:
        print("🚀 Setting up everything...")
        
        # Download/setup models
        download_yolov9_models()
        
        # Create reference data
        create_sample_reference_data()
        
        # Create necessary directories
        for directory in ['logs', 'data']:
            os.makedirs(directory, exist_ok=True)
            print(f"📁 Created directory: {directory}")
        
        print("\n✅ Setup complete!")
        print("\n📋 Next steps:")
        print("1. Replace placeholder model files with real YOLOv9 ONNX models")
        print("2. Update reference_features.json with real feature data")
        print("3. Configure AWS credentials for S3 (if using active learning)")
        print("4. Run: docker-compose up -d")
        print("5. Test: python test_api.py")
        
        return 0
    
    else:
        print("🤔 No action specified. Use --help for options.")
        print("\nQuick start: python setup_models.py --setup-all")
        return 1

if __name__ == "__main__":
    exit(main()) 