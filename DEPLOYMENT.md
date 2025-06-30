# Deployment Guide

This guide covers deploying the parakeetv2API in production environments.

## Prerequisites

### Hardware Requirements

- **GPU**: NVIDIA GPU with CUDA support (recommended: RTX 3080 or better)
- **RAM**: Minimum 16GB (32GB recommended)
- **Storage**: 10GB free space for model cache
- **CPU**: 4+ cores recommended

### Software Requirements

- **Operating System**: Linux (Ubuntu 20.04+ recommended)
- **Python**: 3.11 or higher
- **CUDA**: Version 12.0+ with compatible drivers
- **FFmpeg**: For audio format conversion

## Environment Setup

### 1. System Dependencies

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y build-essential cmake libprotoc-dev protobuf-compiler ffmpeg

# Install NVIDIA drivers and CUDA (if not already installed)
# Follow NVIDIA's official installation guide for your system
```

### 2. Python Environment

```bash
# Create conda environment
conda create -n nemo python=3.11
conda activate nemo

# Install Python dependencies
pip install -r requirements.txt

# For production, also install optional monitoring tools
pip install pynvml  # GPU monitoring
```

### 3. Model Cache Setup

```bash
# Create model cache directory
sudo mkdir -p /var/cache/parakeet
sudo chown $USER:$USER /var/cache/parakeet

# Set environment variable (add to ~/.bashrc for persistence)
export MODEL_CACHE_DIR=/var/cache/parakeet
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Server configuration
HOST=0.0.0.0
PORT=8011
ENVIRONMENT=production

# GPU configuration
GPU_DEVICE=0

# Model configuration
MODEL_CACHE_DIR=/var/cache/parakeet

# Audio processing
MAX_AUDIO_FILE_SIZE=26214400  # 25MB in bytes
TEMP_DIR=/tmp/parakeet

# API configuration
MAX_CONCURRENT_REQUESTS=10
CORS_ORIGINS=["*"]

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# Security (optional)
# API_KEY=your-secret-api-key-here
```

### Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `localhost` | Host to bind server to (`0.0.0.0` for external access) |
| `PORT` | `8011` | Port to listen on |
| `ENVIRONMENT` | `production` | Environment mode (`development`, `staging`, `production`) |
| `GPU_DEVICE` | `None` | GPU device index (0, 1, 2, etc.) |
| `MODEL_CACHE_DIR` | `None` | Directory to cache downloaded models |
| `MAX_AUDIO_FILE_SIZE` | `25MB` | Maximum audio file size in bytes |
| `MAX_CONCURRENT_REQUESTS` | `10` | Maximum concurrent transcription requests |
| `LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

## Deployment Methods

### Method 1: Direct Python Execution

```bash
# Activate environment
conda activate nemo

# Start the server
python -m src.main

# Or use uvicorn directly
uvicorn src.main:app --host 0.0.0.0 --port 8011 --workers 1
```

### Method 2: Systemd Service

Create a systemd service for automatic startup:

```bash
# Create service file
sudo nano /etc/systemd/system/parakeetv2api.service
```

Service file content:

```ini
[Unit]
Description=parakeetv2API Server
After=network.target

[Service]
Type=simple
User=parakeet
Group=parakeet
WorkingDirectory=/opt/parakeetv2API
Environment=PATH=/home/parakeet/miniconda3/envs/nemo/bin
ExecStart=/home/parakeet/miniconda3/envs/nemo/bin/python -m src.main
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

# Resource limits
LimitNOFILE=65536
MemoryMax=16G

# Security settings
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/cache/parakeet /tmp/parakeet /var/log

[Install]
WantedBy=multi-user.target
```

Enable and start the service:

```bash
# Create user for the service
sudo useradd -r -s /bin/false parakeet
sudo mkdir -p /opt/parakeetv2API
sudo chown parakeet:parakeet /opt/parakeetv2API

# Copy application files
sudo cp -r . /opt/parakeetv2API/
sudo chown -R parakeet:parakeet /opt/parakeetv2API

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable parakeetv2api
sudo systemctl start parakeetv2api

# Check status
sudo systemctl status parakeetv2api
```

### Method 3: Docker Deployment

Create a Dockerfile:

```dockerfile
FROM nvidia/cuda:12.1-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-pip \
    python3.11-venv \
    build-essential \
    cmake \
    libprotoc-dev \
    protobuf-compiler \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd -m -u 1000 parakeet
USER parakeet
WORKDIR /app

# Create virtual environment
RUN python3.11 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY --chown=parakeet:parakeet requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=parakeet:parakeet . .

# Expose port
EXPOSE 8011

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8011/health || exit 1

# Start application
CMD ["python", "-m", "src.main"]
```

Build and run:

```bash
# Build image
docker build -t parakeetv2api .

# Run container
docker run -d \
    --name parakeetv2api \
    --gpus all \
    -p 8011:8011 \
    -v /var/cache/parakeet:/var/cache/parakeet \
    -e MODEL_CACHE_DIR=/var/cache/parakeet \
    parakeetv2api
```

## Reverse Proxy Setup

### Nginx Configuration

```nginx
upstream parakeetv2api {
    server 127.0.0.1:8011;
}

server {
    listen 80;
    server_name your-domain.com;

    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    # SSL configuration
    ssl_certificate /path/to/your/certificate.pem;
    ssl_certificate_key /path/to/your/private-key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384;

    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";

    # File upload limits
    client_max_body_size 30M;
    client_body_timeout 60s;

    location / {
        proxy_pass http://parakeetv2api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts for transcription requests
        proxy_connect_timeout 5s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }

    # Health check endpoint (no auth required)
    location /health {
        proxy_pass http://parakeetv2api;
        access_log off;
    }
}
```

## Monitoring and Logging

### Log Management

Configure log rotation:

```bash
# Create logrotate configuration
sudo nano /etc/logrotate.d/parakeetv2api
```

Logrotate configuration:

```
/var/log/parakeetv2api/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 parakeet parakeet
    postrotate
        systemctl reload parakeetv2api
    endscript
}
```

### Prometheus Metrics (Optional)

Add prometheus metrics endpoint by installing:

```bash
pip install prometheus-fastapi-instrumentator
```

### Health Monitoring

Set up monitoring with your preferred tools:

- **Health Check URL**: `http://your-server:8011/health`
- **Metrics**: CPU, memory, GPU usage, response times
- **Alerts**: Model loading failures, high error rates, resource exhaustion

## Security Considerations

### Authentication

Configure API key authentication:

```bash
# Generate a secure API key
openssl rand -hex 32

# Add to .env file
echo "API_KEY=your-generated-key-here" >> .env
```

### Firewall Configuration

```bash
# Allow only necessary ports
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

### File Permissions

```bash
# Secure configuration files
chmod 600 .env
chmod 755 src/
```

## Performance Tuning

### GPU Optimization

```bash
# Set GPU performance mode
sudo nvidia-smi -pm 1

# Set memory clock (adjust for your GPU)
sudo nvidia-smi -ac 5001,1215
```

### System Limits

Edit `/etc/security/limits.conf`:

```
parakeet soft nofile 65536
parakeet hard nofile 65536
parakeet soft nproc 4096
parakeet hard nproc 4096
```

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Check GPU memory availability
   - Verify CUDA installation
   - Check model cache permissions

2. **Audio Processing Failures**
   - Verify FFmpeg installation
   - Check file format support
   - Review file size limits

3. **Memory Issues**
   - Monitor GPU memory usage
   - Adjust `MAX_CONCURRENT_REQUESTS`
   - Consider model quantization

### Log Analysis

```bash
# View service logs
sudo journalctl -u parakeetv2api -f

# Check GPU usage
nvidia-smi

# Monitor system resources
htop
```

### Health Check Commands

```bash
# Basic health check
curl http://localhost:8011/health

# Test transcription
curl -X POST http://localhost:8011/v1/audio/transcriptions \
  -F file="@test.wav" \
  -F model="whisper-1"
```

## Backup and Recovery

### Model Cache Backup

```bash
# Backup model cache
tar -czf model-cache-backup.tar.gz /var/cache/parakeet/

# Restore model cache
tar -xzf model-cache-backup.tar.gz -C /
```

### Configuration Backup

```bash
# Backup configuration
cp .env config-backup.env
```

## Scaling

### Load Balancing

For high-traffic deployments, consider:

1. **Multiple Instances**: Run multiple API instances behind a load balancer
2. **GPU Sharing**: Use NVIDIA MPS for GPU sharing
3. **Queue Management**: Implement request queuing for better resource management

### Resource Planning

- **CPU**: 1 core per concurrent request
- **Memory**: 2GB base + 1GB per concurrent request
- **GPU Memory**: 6-8GB for the model + overhead
- **Storage**: 10GB for model cache + logs