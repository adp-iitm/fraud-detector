#!/usr/bin/env python3
import uvicorn
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    print("Starting Fraud & Phishing Detection API...")
    print("API Documentation: http://localhost:8000/docs")
    print("Health Check: http://localhost:8000/health")
    print()
    
    try:
        uvicorn.run(
            "ml_models.api.endpoints:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please install dependencies: pip install fastapi uvicorn")
    except Exception as e:
        print(f"Error starting API: {e}")
