# Docker Setup for parakeetv2API

This guide explains how to run parakeetv2API using Docker with GPU support.

## Prerequisites

- Docker Engine with Docker Compose
- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit (nvidia-docker2)

## Quick Start

### Build and Run with Default GPU (GPU 0)

```bash
docker-compose up --build
```

The API will be accessible at:
- From the host machine: `http://localhost:8011`
- From other machines on the LAN: `http://<host-ip>:8011`

### Specify Different GPU

You can specify which GPU to use by setting the `NVIDIA_VISIBLE_DEVICES` environment variable:

```bash
# Use GPU 1
NVIDIA_VISIBLE_DEVICES=1 docker-compose up

# Use GPUs 0 and 2
NVIDIA_VISIBLE_DEVICES=0,2 docker-compose up

# Use all available GPUs
NVIDIA_VISIBLE_DEVICES=all docker-compose up
```

### Run in Background

```bash
docker-compose up -d
```

### View Logs

```bash
docker-compose logs -f
```

### Stop the Service

```bash
docker-compose down
```

## Alternative: Using Docker Directly

If you prefer not to use docker-compose:

```bash
# Build the image
docker build -t parakeetv2api .

# Run with default GPU (0)
docker run -d --gpus '"device=0"' -p 8011:8011 --name parakeetv2api parakeetv2api

# Run with specific GPU
docker run -d --gpus '"device=1"' -p 8011:8011 --name parakeetv2api parakeetv2api

# Run with multiple GPUs
docker run -d --gpus '"device=0,2"' -p 8011:8011 --name parakeetv2api parakeetv2api
```

## Testing the API

Once running, the API will be available at `http://localhost:8011` (or `http://<host-ip>:8011` from other machines).

### From the Host Machine

```bash
# Check available models
curl http://localhost:8011/v1/models

# Transcribe audio
curl -X POST http://localhost:8011/v1/audio/transcriptions \
  -F "file=@path/to/audio.wav" \
  -F "model=whisper-1"
```

### From Other Machines on the LAN

Replace `<host-ip>` with the actual IP address of the machine running Docker:

```bash
# Find the host IP address (on the Docker host)
hostname -I | awk '{print $1}'

# From another machine, check available models
curl http://<host-ip>:8011/v1/models

# From another machine, transcribe audio
curl -X POST http://<host-ip>:8011/v1/audio/transcriptions \
  -F "file=@path/to/audio.wav" \
  -F "model=whisper-1"
```

## Network Access

The service is configured to accept connections from any network interface (`0.0.0.0:8011`), allowing access from:
- The Docker host itself (`localhost`)
- Other machines on the same LAN
- Any reachable network (be cautious in production environments)

To restrict access to localhost only, modify the port binding in `docker-compose.yml`:
```yaml
ports:
  - "127.0.0.1:8011:8011"
```

## Troubleshooting

### GPU Not Available

If you get GPU-related errors, ensure:

1. NVIDIA drivers are installed: `nvidia-smi`
2. NVIDIA Container Toolkit is installed: `docker run --rm --gpus all nvidia/cuda:12.9.1-base-ubuntu24.04 nvidia-smi`
3. Docker daemon is restarted after installing nvidia-docker2

### Permission Denied

If you get permission errors, ensure your user is in the docker group:

```bash
sudo usermod -aG docker $USER
# Log out and back in for changes to take effect
```

### Connection Refused from LAN

If other machines cannot connect:

1. Check the host firewall allows port 8011:
   ```bash
   # Ubuntu/Debian
   sudo ufw allow 8011
   
   # RHEL/CentOS
   sudo firewall-cmd --permanent --add-port=8011/tcp
   sudo firewall-cmd --reload
   ```

2. Verify the service is running:
   ```bash
   docker-compose ps
   curl http://localhost:8011/v1/models
   ```

3. Confirm the correct IP address:
   ```bash
   ip addr show | grep inet
   ```