#!/usr/bin/env python3
"""
Test script for Object Detection API
"""

import asyncio
import aiohttp
import requests
import json
import time
from pathlib import Path
from typing import Dict, Any

# Test configuration
API_BASE_URL = "http://localhost:8000"
TEST_IMAGE_PATH = "test_image.jpg"  # You need to provide this

async def test_health_check():
    """Test health check endpoint"""
    print("🔍 Testing health check...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        response.raise_for_status()
        
        data = response.json()
        print(f"✅ Health check passed: {data}")
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

async def test_metrics_endpoint():
    """Test Prometheus metrics endpoint"""
    print("📊 Testing metrics endpoint...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/metrics")
        response.raise_for_status()
        
        metrics_text = response.text
        print(f"✅ Metrics endpoint accessible (length: {len(metrics_text)} chars)")
        
        # Check for key metrics
        required_metrics = [
            "object_detection_requests_total",
            "object_detection_request_duration_seconds",
            "object_detection_inference_duration_seconds"
        ]
        
        for metric in required_metrics:
            if metric in metrics_text:
                print(f"  ✅ Found metric: {metric}")
            else:
                print(f"  ⚠️ Missing metric: {metric}")
        
        return True
    except Exception as e:
        print(f"❌ Metrics endpoint failed: {e}")
        return False

async def test_prediction_endpoint(version: str = "v1"):
    """Test prediction endpoint"""
    print(f"🧠 Testing prediction endpoint {version}...")
    
    if not Path(TEST_IMAGE_PATH).exists():
        print(f"⚠️ Test image not found: {TEST_IMAGE_PATH}")
        print("Creating a dummy image for testing...")
        
        # Create a small test image
        try:
            from PIL import Image
            import numpy as np
            
            # Create a 640x480 RGB image
            img_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(TEST_IMAGE_PATH)
            print(f"✅ Created test image: {TEST_IMAGE_PATH}")
        except ImportError:
            print("❌ PIL not available, cannot create test image")
            return False
    
    try:
        # Test prediction
        with open(TEST_IMAGE_PATH, "rb") as f:
            files = {"file": f}
            data = {"conf_threshold": 0.5}
            
            start_time = time.time()
            response = requests.post(f"{API_BASE_URL}/{version}/predict", files=files, data=data)
            end_time = time.time()
            
            response.raise_for_status()
            
            result = response.json()
            request_time = end_time - start_time
            
            print(f"✅ Prediction successful ({version})")
            print(f"  📊 Response time: {request_time:.3f}s")
            print(f"  🎯 Detections found: {len(result.get('detections', []))}")
            print(f"  ⚡ Inference time: {result.get('inference_time', 0):.3f}s")
            print(f"  🆔 Request ID: {result.get('request_id', 'N/A')}")
            
            # Print first few detections
            detections = result.get('detections', [])
            if detections:
                print("  🔍 Sample detections:")
                for i, det in enumerate(detections[:3]):
                    print(f"    {i+1}. {det.get('class_name', 'unknown')} "
                          f"(confidence: {det.get('confidence', 0):.3f})")
            
            return True
            
    except Exception as e:
        print(f"❌ Prediction failed ({version}): {e}")
        return False

async def test_drift_endpoints():
    """Test drift detection endpoints"""
    print("📈 Testing drift detection endpoints...")
    
    try:
        # Test drift status
        response = requests.get(f"{API_BASE_URL}/drift/status")
        response.raise_for_status()
        
        drift_status = response.json()
        print(f"✅ Drift status retrieved:")
        print(f"  📊 Buffer size: {drift_status.get('buffer_size', 0)}")
        print(f"  📋 Has reference data: {drift_status.get('has_reference_data', False)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Drift endpoints failed: {e}")
        return False

async def load_test(version: str = "v1", num_requests: int = 10):
    """Perform basic load testing"""
    print(f"🚀 Load testing {version} endpoint with {num_requests} requests...")
    
    if not Path(TEST_IMAGE_PATH).exists():
        print(f"⚠️ Test image not found: {TEST_IMAGE_PATH}")
        return False
    
    async def make_request(session, request_id):
        try:
            with open(TEST_IMAGE_PATH, "rb") as f:
                data = aiohttp.FormData()
                data.add_field("file", f, filename="test.jpg", content_type="image/jpeg")
                data.add_field("conf_threshold", "0.5")
                
                start_time = time.time()
                async with session.post(f"{API_BASE_URL}/{version}/predict", data=data) as response:
                    end_time = time.time()
                    
                    if response.status == 200:
                        result = await response.json()
                        return {
                            "success": True,
                            "response_time": end_time - start_time,
                            "inference_time": result.get("inference_time", 0),
                            "detections": len(result.get("detections", []))
                        }
                    else:
                        return {"success": False, "status": response.status}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # Run concurrent requests
    async with aiohttp.ClientSession() as session:
        tasks = [make_request(session, i) for i in range(num_requests)]
        results = await asyncio.gather(*tasks)
    
    # Analyze results
    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]
    
    if successful:
        response_times = [r["response_time"] for r in successful]
        inference_times = [r["inference_time"] for r in successful]
        
        print(f"✅ Load test completed:")
        print(f"  📊 Successful requests: {len(successful)}/{num_requests}")
        print(f"  ⚡ Average response time: {sum(response_times)/len(response_times):.3f}s")
        print(f"  🧠 Average inference time: {sum(inference_times)/len(inference_times):.3f}s")
        print(f"  ⏱️ Min/Max response time: {min(response_times):.3f}s / {max(response_times):.3f}s")
        
        if failed:
            print(f"  ❌ Failed requests: {len(failed)}")
            for fail in failed[:3]:  # Show first 3 failures
                print(f"    - {fail}")
    else:
        print(f"❌ All requests failed")
        for fail in failed[:5]:  # Show first 5 failures
            print(f"  - {fail}")
    
    return len(successful) > 0

async def main():
    """Main test function"""
    print("🧪 Starting Object Detection API Tests\n")
    
    tests = [
        ("Health Check", test_health_check()),
        ("Metrics Endpoint", test_metrics_endpoint()),
        ("Prediction V1", test_prediction_endpoint("v1")),
        ("Prediction V2", test_prediction_endpoint("v2")),
        ("Drift Detection", test_drift_endpoints()),
    ]
    
    results = []
    for test_name, test_coro in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print('='*50)
        
        try:
            result = await test_coro
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Load test
    print(f"\n{'='*50}")
    print("Running: Load Test")
    print('='*50)
    
    try:
        load_result = await load_test("v1", 5)
        results.append(("Load Test V1", load_result))
    except Exception as e:
        print(f"❌ Load test crashed: {e}")
        results.append(("Load Test V1", False))
    
    # Summary
    print(f"\n{'='*50}")
    print("📋 TEST SUMMARY")
    print('='*50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! API is working correctly.")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    asyncio.run(main()) 