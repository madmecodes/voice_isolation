FROM tensorflow/tensorflow:latest-gpu

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Install specific versions of spleeter and other dependencies
RUN pip install --no-cache-dir spleeter==2.4.2 tensorflow-io-gcs-filesystem==0.32.0 psutil
RUN pip install --no-cache-dir nvidia-ml-py3

# Copy your script
COPY voice_isolation.py .

# Make a directory for audio files
RUN mkdir -p /audio

# Set volume
VOLUME ["/audio"]

# Command to run
CMD ["python", "voice_isolation.py"]