# Dockerfile SIMPLE pour AuraSR V2
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Install AuraSR
RUN pip install aura-sr runpod

# Simple handler
COPY <<EOF handler.py
import runpod
from aura_sr import AuraSR
from PIL import Image
import requests
from io import BytesIO
import base64

# Load model
print("Loading AuraSR V2...")
aura_sr = AuraSR.from_pretrained("fal/AuraSR-v2")
print("Model loaded!")

def handler(job):
    try:
        job_input = job.get("input", {})
        image_url = job_input.get("image")
        
        if not image_url:
            return {"error": "No image URL provided"}
        
        # Download image
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        
        # Upscale
        upscaled = aura_sr.upscale_4x_overlapped(image)
        
        # Convert to base64
        buffer = BytesIO()
        upscaled.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return {
            "status": "success",
            "upscaled_image": img_base64,
            "original_size": image.size,
            "upscaled_size": upscaled.size
        }
        
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
EOF

CMD ["python", "handler.py"]
