.PHONY: cpu-build cpu-run gpu-build gpu-run clean help

help:
	@echo "Voice Isolation Tool - Command Reference"
	@echo ""
	@echo "Available commands:"
	@echo "  make cpu-build  - Build the CPU container"
	@echo "  make cpu-run    - Run the CPU version (for smaller files)"
	@echo "  make gpu-build  - Build the GPU container"
	@echo "  make gpu-run    - Run the GPU version (for larger files)"
	@echo "  make clean      - Remove temporary files"
	@echo "  make help       - Show this help message"
	@echo ""
	@echo "Usage example:"
	@echo "  1. Place audio files in the 'audio' directory"
	@echo "  2. make gpu-run (for large files) or make cpu-run (for small files)"

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

clean:
	@echo "Cleaning temporary files..."
	find . -name "spleeter_temp" -type d -exec rm -rf {} +
	find . -name "__pycache__" -type d -exec rm -rf {} +
	@echo "Clean completed!"