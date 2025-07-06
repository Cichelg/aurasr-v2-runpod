# Dockerfile optimisé pour AuraSR V2 sur RunPod
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install AuraSR package
RUN pip install --no-cache-dir aura-sr

# Copy custom gradio extension
COPY gradio_imageslider-0.0.20-py3-none-any.whl .
RUN pip install --no-cache-dir gradio_imageslider-0.0.20-py3-none-any.whl

# Copy application files
COPY app.py .
COPY README.md .

# Create RunPod handler wrapper
COPY <<EOF handler.py
import runpod
import gradio as gr
import os
import requests
import json
from threading import Thread
import time

# Import your Gradio app
from app import demo

class GradioHandler:
    def __init__(self):
        self.gradio_port = 7860
        self.gradio_process = None
        self.start_gradio_server()
        
    def start_gradio_server(self):
        """Start Gradio server in background thread"""
        def run_gradio():
            demo.launch(
                server_name="0.0.0.0",
                server_port=self.gradio_port,
                share=False,
                show_error=True,
                quiet=True
            )
        
        self.gradio_process = Thread(target=run_gradio, daemon=True)
        self.gradio_process.start()
        
        # Wait for server to be ready
        time.sleep(10)
        
    def process_image(self, image_data):
        """Process image through Gradio API"""
        try:
            # Prepare the request to local Gradio server
            url = f"http://localhost:{self.gradio_port}/api/predict"
            
            # Convert base64 image to file-like object
            import base64
            from io import BytesIO
            from PIL import Image
            
            if isinstance(image_data, str):
                # If it's a base64 string
                image_bytes = base64.b64decode(image_data)
                image = Image.open(BytesIO(image_bytes))
            else:
                # If it's already an image
                image = image_data
                
            # Save temporarily
            temp_path = "/tmp/input_image.png"
            image.save(temp_path)
            
            # Call Gradio API
            with open(temp_path, "rb") as f:
                files = {"data": f}
                response = requests.post(url, files=files)
                
            if response.status_code == 200:
                result = response.json()
                return result
            else:
                return {"error": f"Gradio API error: {response.status_code}"}
                
        except Exception as e:
            return {"error": f"Processing error: {str(e)}"}

# Initialize handler
handler_instance = GradioHandler()

def runpod_handler(job):
    """RunPod serverless handler function"""
    try:
        job_input = job.get("input", {})
        
        # Get image from job input
        if "image" in job_input:
            image_data = job_input["image"]
            result = handler_instance.process_image(image_data)
            return {"status": "success", "output": result}
        else:
            return {"error": "No image provided in input"}
            
    except Exception as e:
        return {"error": f"Handler error: {str(e)}"}

if __name__ == "__main__":
    # Check if running in RunPod environment
    if os.getenv("RUNPOD_ENDPOINT_ID"):
        # Start RunPod serverless worker
        runpod.serverless.start({"handler": runpod_handler})
    else:
        # Run Gradio app directly for local testing
        from app import demo
        demo.launch(server_name="0.0.0.0", server_port=7860)
EOF

# Expose ports
EXPOSE 7860 8080

# Set environment variables
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# Default command
CMD ["python", "handler.py"]
