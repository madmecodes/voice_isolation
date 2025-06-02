# Better Use Google Colab

https://colab.research.google.com/drive/1Ss849YhhpBCejdGxh07-PrTJbplTtYTD?usp=sharing

# Run Locally Via Docker

A Docker-based tool for isolating vocals from audio files using Spleeter, with support for both CPU and GPU processing.

## Features

- Isolate vocals from any audio file (MP3, WAV, FLAC, OGG)
- CPU version for quick processing of small files
- GPU version for faster processing of large files
- Detailed logging and error handling
- Performance monitoring and timing statistics

## Requirements

- Docker and Docker Compose
- For GPU acceleration:
  - NVIDIA GPU with compatible drivers (for NVIDIA GPUs)
  - Apple Silicon M1/M2/M3 (for Mac)
  - NVIDIA Container Toolkit (for NVIDIA GPUs)

## Directory Structure

```
voice-isolation/
├── cpu/                   # CPU-only implementation
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── voice_isolation.py
├── gpu/                   # GPU-accelerated implementation
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── voice_isolation.py
├── audio/                 # Shared audio directory
└── Makefile               # Easy setup commands
```

## Quick Start

### Step 1: Clone the repository

```bash
git clone https://github.com/yourusername/voice-isolation.git
cd voice-isolation
```

### Step 2: Place your audio files in the audio directory

```bash
cp /path/to/your/audio.mp3 audio/
```

### Step 3: Choose CPU or GPU version

For smaller files (under 10 minutes):
```bash
make cpu-build
make cpu-run
```

For larger files (over 10 minutes):
```bash
make gpu-build
make gpu-run
```

### Step 4: Process your file

When prompted, enter the path as:
```
/audio/your-file.mp3
```

The processed file will be saved as:
```
/audio/your-file_isolated.wav
```

## Performance

Processing times vary based on file size and hardware:

| File Duration | CPU Processing | GPU Processing (Approx) |
|---------------|---------------|------------------------|
| 3 minutes     | ~3-5 minutes  | ~30-60 seconds         |
| 45 minutes    | ~45-90 minutes| ~10-20 minutes         |

## GPU Configuration

### NVIDIA GPUs
Make sure you have the NVIDIA Container Toolkit installed:
```bash
sudo apt-get install nvidia-container-toolkit
sudo systemctl restart docker
```

### Apple Silicon (M1/M2/M3)
The GPU version should automatically detect and use the Metal backend for TensorFlow.

## Troubleshooting

If you encounter issues:

1. Check the log files in the audio directory
2. Ensure you have the right permissions for file access
3. For GPU version, verify your GPU is properly detected
4. Try the CPU version if the GPU version has compatibility issues

## License

[MIT License](LICENSE)

## Acknowledgements

This tool uses [Spleeter](https://github.com/deezer/spleeter) by Deezer Research for audio source separation.