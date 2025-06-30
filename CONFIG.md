# Configuration Reference

This document provides a comprehensive reference for all configuration options available in parakeetv2API.

## Configuration Methods

Configuration can be provided through:

1. **Environment Variables** (recommended for production)
2. **`.env` file** (recommended for development)
3. **Direct environment setup**

## Server Configuration

### Network Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `HOST` | string | `localhost` | Host address to bind the server to |
| `PORT` | integer | `8011` | Port number to listen on |
| `API_PREFIX` | string | `/v1` | API route prefix |

**Examples:**
```bash
# Listen on all interfaces
HOST=0.0.0.0

# Use custom port
PORT=9000

# Custom API prefix
API_PREFIX=/api/v1
```

**Valid HOST values:**
- `localhost` - Local access only
- `127.0.0.1` - Local access only
- `0.0.0.0` - All interfaces (external access)
- `192.168.x.x` - Local network access
- `10.x.x.x` - Local network access

### CORS Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `CORS_ORIGINS` | list | `["*"]` | Allowed CORS origins |

**Examples:**
```bash
# Allow all origins (development only)
CORS_ORIGINS=["*"]

# Specific domains only
CORS_ORIGINS=["https://myapp.com", "https://api.myapp.com"]

# Local development
CORS_ORIGINS=["http://localhost:3000", "http://127.0.0.1:3000"]
```

## GPU Configuration

### Device Selection

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `GPU_DEVICE` | integer | `None` | Specific GPU device index to use |

**Examples:**
```bash
# Use first GPU
GPU_DEVICE=0

# Use second GPU
GPU_DEVICE=1

# Auto-select (default)
# GPU_DEVICE=
```

**Note:** If not specified, CUDA will automatically select an available GPU.

## Model Configuration

### Model Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MODEL_NAME` | string | `nvidia/parakeet-tdt-0.6b-v2` | ASR model identifier |
| `MODEL_CACHE_DIR` | string | `None` | Directory to cache downloaded models |

**Examples:**
```bash
# Default model (recommended)
MODEL_NAME=nvidia/parakeet-tdt-0.6b-v2

# Custom cache directory
MODEL_CACHE_DIR=/var/cache/parakeet

# Use system temporary directory (default)
# MODEL_CACHE_DIR=
```

**Model Cache Behavior:**
- If `MODEL_CACHE_DIR` is not set, models are cached in the system's temporary directory
- The first run will download the model (~2GB)
- Subsequent runs will use the cached model for faster startup

## Audio Processing Configuration

### File Handling

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MAX_AUDIO_FILE_SIZE` | integer | `26214400` | Maximum audio file size in bytes (25MB) |
| `TEMP_DIR` | string | `None` | Directory for temporary audio files |

**Examples:**
```bash
# 50MB limit
MAX_AUDIO_FILE_SIZE=52428800

# 10MB limit
MAX_AUDIO_FILE_SIZE=10485760

# Custom temp directory
TEMP_DIR=/tmp/parakeet

# Use system temp (default)
# TEMP_DIR=
```

**Supported Audio Formats:**
- WAV (`.wav`) - Recommended
- MP3 (`.mp3`)
- FLAC (`.flac`)
- M4A (`.m4a`)
- OGG (`.ogg`)
- MP4 (`.mp4`, `.mpeg`, `.mpga`)
- WebM (`.webm`)

## Performance Configuration

### Concurrency Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MAX_CONCURRENT_REQUESTS` | integer | `10` | Maximum concurrent transcription requests |

**Examples:**
```bash
# High-performance server
MAX_CONCURRENT_REQUESTS=20

# Low-resource environment
MAX_CONCURRENT_REQUESTS=5

# Single request processing
MAX_CONCURRENT_REQUESTS=1
```

**Performance Notes:**
- Higher concurrency requires more GPU memory
- Monitor GPU memory usage when increasing this value
- Each request typically uses 500MB-1GB of GPU memory

## Logging Configuration

### Log Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `LOG_LEVEL` | string | `INFO` | Logging verbosity level |
| `LOG_FORMAT` | string | `json` | Log output format |
| `ENVIRONMENT` | string | `production` | Environment mode |

**Log Levels:**
- `DEBUG` - Detailed debugging information
- `INFO` - General information messages
- `WARNING` - Warning messages
- `ERROR` - Error messages only
- `CRITICAL` - Critical errors only

**Log Formats:**
- `json` - Structured JSON logs (recommended for production)
- `text` - Human-readable text logs (development)

**Environment Modes:**
- `development` - Development settings (pretty logs, debug enabled)
- `staging` - Staging environment settings
- `production` - Production settings (JSON logs, monitoring enabled)

**Examples:**
```bash
# Development setup
ENVIRONMENT=development
LOG_LEVEL=DEBUG
LOG_FORMAT=text

# Production setup
ENVIRONMENT=production
LOG_LEVEL=INFO
LOG_FORMAT=json

# Debugging issues
LOG_LEVEL=DEBUG
```

## Development Configuration

### Development Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DEBUG` | boolean | `False` | Enable debug mode |
| `RELOAD` | boolean | `False` | Enable auto-reload for development |

**Examples:**
```bash
# Development mode
DEBUG=true
RELOAD=true

# Production mode (default)
DEBUG=false
RELOAD=false
```

**Debug Mode Effects:**
- Enables detailed error messages
- Exposes API documentation at `/docs`
- Includes stack traces in error responses
- **Never enable in production**

## Security Configuration

### Authentication (Optional)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `API_KEY` | string | `None` | API key for authentication |

**Examples:**
```bash
# Enable authentication
API_KEY=your-secret-api-key-here

# No authentication (default)
# API_KEY=
```

**Authentication Behavior:**
- If `API_KEY` is set, all requests must include `Authorization: Bearer <API_KEY>` header
- If not set, all requests are allowed (useful for internal services)
- Health check endpoints (`/` and `/health`) are never protected

## Complete Configuration Examples

### Development Configuration

`.env` file for development:
```bash
# Server
HOST=localhost
PORT=8011
DEBUG=true
RELOAD=true
ENVIRONMENT=development

# Logging
LOG_LEVEL=DEBUG
LOG_FORMAT=text

# Performance
MAX_CONCURRENT_REQUESTS=3

# GPU (if available)
GPU_DEVICE=0

# Directories
MODEL_CACHE_DIR=./cache
TEMP_DIR=./tmp
```

### Production Configuration

`.env` file for production:
```bash
# Server
HOST=0.0.0.0
PORT=8011
DEBUG=false
ENVIRONMENT=production

# Security
API_KEY=your-production-api-key-here
CORS_ORIGINS=["https://yourdomain.com"]

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# Performance
MAX_CONCURRENT_REQUESTS=10
MAX_AUDIO_FILE_SIZE=26214400

# GPU
GPU_DEVICE=0

# Directories
MODEL_CACHE_DIR=/var/cache/parakeet
TEMP_DIR=/tmp/parakeet
```

### Docker Configuration

Environment variables for Docker:
```bash
# Use environment variables instead of .env file
docker run -d \
  --name parakeetv2api \
  --gpus all \
  -p 8011:8011 \
  -e HOST=0.0.0.0 \
  -e PORT=8011 \
  -e ENVIRONMENT=production \
  -e LOG_LEVEL=INFO \
  -e MAX_CONCURRENT_REQUESTS=10 \
  -e MODEL_CACHE_DIR=/var/cache/parakeet \
  -v /host/cache:/var/cache/parakeet \
  parakeetv2api
```

## Validation Rules

### Host Validation
```bash
# Valid hosts
HOST=localhost          # ✓
HOST=127.0.0.1         # ✓
HOST=0.0.0.0           # ✓
HOST=192.168.1.100     # ✓
HOST=10.0.0.50         # ✓

# Invalid hosts
HOST=example.com       # ✗ (not allowed for security)
HOST=8.8.8.8          # ✗ (public IP not allowed)
```

### Port Validation
```bash
# Valid ports
PORT=8011              # ✓
PORT=80                # ✓
PORT=443               # ✓
PORT=9000              # ✓

# Invalid ports
PORT=0                 # ✗ (reserved)
PORT=70000             # ✗ (out of range)
PORT=-1                # ✗ (negative)
```

### Log Level Validation
```bash
# Valid log levels (case insensitive)
LOG_LEVEL=DEBUG        # ✓
LOG_LEVEL=info         # ✓
LOG_LEVEL=Warning      # ✓

# Invalid log levels
LOG_LEVEL=VERBOSE      # ✗ (not a valid level)
LOG_LEVEL=1            # ✗ (numeric not allowed)
```

## Environment Variable Priority

Configuration is loaded in this order (later values override earlier ones):

1. Default values (defined in code)
2. `.env` file in project root
3. System environment variables
4. Command-line arguments (if applicable)

**Example:**
```bash
# .env file
PORT=8000

# Environment variable (overrides .env)
export PORT=9000

# Application will use PORT=9000
```

## Configuration Validation

The application validates all configuration on startup:

- **Type checking**: Ensures integers are integers, booleans are booleans
- **Range validation**: Ports must be 1-65535, file sizes must be positive
- **Path validation**: Directories must be writable if specified
- **Format validation**: Log levels must be valid options

If configuration is invalid, the application will:
1. Log the specific validation error
2. Show the invalid value and expected format
3. Exit with a non-zero status code

## Troubleshooting Configuration

### Common Issues

1. **"Address already in use"**
   - Another service is using the configured port
   - Check with `netstat -tlnp | grep :8011`
   - Change `PORT` to an available port

2. **"Permission denied" on directories**
   - The specified directory is not writable
   - Check permissions: `ls -la /path/to/directory`
   - Fix with: `chmod 755 /path/to/directory`

3. **"Invalid host" error**
   - The specified host is not allowed
   - Use `localhost`, `127.0.0.1`, `0.0.0.0`, or local network IPs

4. **GPU not found**
   - Check GPU availability: `nvidia-smi`
   - Verify CUDA installation
   - Try `GPU_DEVICE=` to auto-select

### Debugging Configuration

To see the active configuration:

```bash
# Enable debug logging to see loaded config
LOG_LEVEL=DEBUG python -m src.main
```

The application logs all configuration values on startup (sensitive values are masked).