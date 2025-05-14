.PHONY: cpu-build cpu-run gpu-build gpu-run clean help

help:
	@echo "Voice Isolation Tool - Command Reference"
	@echo ""
	@echo "Available commands:"
	@echo "  make cpu-build        - Build the CPU container"
	@echo "  make cpu-run          - Run the CPU version (for smaller files)"
	@echo "  make gpu-build        - Build the GPU container"
	@echo "  make gpu-run          - Run the GPU version (for larger files)"
	@echo "  make cpu-batch-build  - Build the CPU batch processing container"
	@echo "  make cpu-batch-run    - Run the CPU batch processing version (for files over 10 min)"
	@echo "  make gpu-batch-build  - Build the GPU batch processing container"
	@echo "  make gpu-batch-run    - Run the GPU batch processing version (for files over 10 min)"
	@echo "  make clean            - Remove temporary files"
	@echo "  make help             - Show this help message"
	@echo ""
	@echo "Usage example:"
	@echo "  1. Place audio files in the 'audio' directory"
	@echo "  2. For regular files: make gpu-run (for large files) or make cpu-run (for small files)"
	@echo "  3. For files over 10 minutes: make gpu-batch-run (recommended) or make cpu-batch-run"

cpu-build:
	@echo "Building CPU container..."
	cd cpu && docker-compose build

cpu-run:
	@echo "Running CPU version..."
	cd cpu && docker-compose run --rm voice-isolation

gpu-build:
	@echo "Building GPU container..."
	cd gpu && docker-compose build

gpu-run:
	@echo "Running GPU version..."
	cd gpu && docker-compose run --rm voice-isolation

cpu-batch-build:
	@echo "Building CPU batch processing container..."
	cd cpu && docker build -f Dockerfile.batch -t voice-isolation-batch-cpu .

cpu-batch-run:
	@echo "Running CPU batch version..."
	cd cpu && docker run --rm -it -v $(PWD)/audio:/audio voice-isolation-batch-cpu

gpu-batch-build:
	@echo "Building GPU batch processing container..."
	cd gpu && docker build -f Dockerfile.batch -t voice-isolation-batch-gpu .

gpu-batch-run:
	@echo "Running GPU batch version..."
	cd gpu && docker run --rm -it --gpus all -v $(PWD)/audio:/audio voice-isolation-batch-gpu

clean:
	@echo "Cleaning temporary files..."
	find . -name "spleeter_temp" -type d -exec rm -rf {} +
	find . -name "__pycache__" -type d -exec rm -rf {} +
	@echo "Clean completed!"