FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Install specific versions of tensorflow and spleeter
RUN pip install --no-cache-dir tensorflow==2.12.1 tensorflow-io-gcs-filesystem==0.32.0 spleeter==2.4.2 psutil

# Copy your script
COPY voice_isolation.py .

# Make a directory for audio files
RUN mkdir -p /audio

# Set volume
VOLUME ["/audio"]

# Command to run
CMD ["python", "voice_isolation.py"]