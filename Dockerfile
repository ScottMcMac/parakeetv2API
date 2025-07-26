# Use NVIDIA CUDA base image with Python
FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set Python as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Set working directory
WORKDIR /app

# Create virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the API port
EXPOSE 8011

# Set environment variables for GPU usage
# Default to GPU 0, can be overridden at runtime
ENV NVIDIA_VISIBLE_DEVICES=0
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Run the FastAPI application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8011"]