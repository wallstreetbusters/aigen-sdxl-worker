# GPU-ready PyTorch image from RunPod
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

# Where our code lives inside the container
WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the repo (handler.py, etc.)
COPY . .

# Start the RunPod serverless worker
CMD ["python", "-u", "handler.py"]
