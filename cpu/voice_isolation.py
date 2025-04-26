import os
import sys
import time
import logging
import traceback
from pathlib import Path
from spleeter.separator import Separator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/audio/voice_isolation_cpu.log')
    ]
)
logger = logging.getLogger(__name__)

def validate_file(file_path):
    """Check if the input file exists and is a supported audio format."""
    supported_formats = {'.mp3', '.wav', '.flac', '.ogg'}
    if not os.path.isfile(file_path):
        return False, "Error: File does not exist."
    if not any(file_path.lower().endswith(fmt) for fmt in supported_formats):
        return False, "Error: Unsupported file format. Use MP3, WAV, FLAC, or OGG."
    return True, ""

def format_time(seconds):
    """Format seconds into hours, minutes, and seconds."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"

def get_file_size_mb(file_path):
    """Get file size in MB."""
    return os.path.getsize(file_path) / (1024 * 1024)

def main():
    logger.info("=== Voice Isolation Tool (Powered by Spleeter) - CPU Version ===")
    print("=== Voice Isolation Tool (Powered by Spleeter) - CPU Version ===")
    print("This script isolates vocals from an audio file, removing background noise.")
    print("Running in Docker container with compatible dependencies.")
    logger.info("Tool started in Docker container")
    
    print("\nInstructions:")
    print("- Enter the full path to your audio file (e.g., /audio/interview.mp3).")
    print("- Make sure your audio files are in the 'audio' directory that's mounted to the container.")
    print("- Output will be saved as '<filename>_isolated.wav' in the same directory.")
    print("- Logs are saved to /audio/voice_isolation_cpu.log")
    print("\nType 'exit' to quit at any time.")

    while True:
        # Prompt for input file
        input_path = input("\nEnter the audio file path (or 'exit' to quit): ").strip()

        if input_path.lower() == 'exit':
            logger.info("User requested exit")
            print("Exiting the tool. Goodbye!")
            sys.exit(0)

        logger.info(f"User entered path: {input_path}")

        # Validate input file
        is_valid, error_msg = validate_file(input_path)
        if not is_valid:
            logger.error(f"Invalid file: {error_msg}")
            print(error_msg)
            continue

        try:
            # Log file details
            file_size_mb = get_file_size_mb(input_path)
            logger.info(f"Processing file: {input_path} (Size: {file_size_mb:.2f} MB)")
            
            # Initialize Spleeter
            print("Initializing Spleeter...")
            logger.info("Initializing Spleeter")
            separator = Separator('spleeter:2stems')

            # Get file details
            input_path = Path(input_path)
            input_dir = input_path.parent
            input_name = input_path.stem
            output_file = input_dir / f"{input_name}_isolated.wav"

            # Create a temporary output directory
            temp_output_dir = input_dir / "spleeter_temp"
            os.makedirs(temp_output_dir, exist_ok=True)
            logger.info(f"Created temporary directory: {temp_output_dir}")

            # Process audio with timing
            print(f"Processing '{input_path.name}' to isolate vocals...")
            print(f"File size: {file_size_mb:.2f} MB - This may take a while for larger files.")
            print("Processing... (Check logs for details)")
            
            if file_size_mb > 50:
                print("\n⚠️ NOTICE: This is a large file. For files over 50MB, consider using the GPU version for faster processing.")
                logger.warning(f"Large file detected: {file_size_mb:.2f} MB. CPU processing may be slow.")
            
            start_time = time.time()
            logger.info(f"Starting separation process for {input_path.name}")
            
            # Memory usage before processing
            try:
                import psutil
                process = psutil.Process(os.getpid())
                mem_before = process.memory_info().rss / 1024 / 1024
                logger.info(f"Memory usage before processing: {mem_before:.2f} MB")
            except ImportError:
                logger.info("psutil not available, skipping memory usage tracking")
            
            try:
                separator.separate_to_file(str(input_path), str(temp_output_dir))
                logger.info("Separation completed successfully")
            except Exception as sep_error:
                logger.error(f"Separation failed: {str(sep_error)}")
                logger.error(traceback.format_exc())
                raise
            
            # Memory usage after processing
            try:
                import psutil
                mem_after = process.memory_info().rss / 1024 / 1024
                logger.info(f"Memory usage after processing: {mem_after:.2f} MB")
                logger.info(f"Memory increase: {mem_after - mem_before:.2f} MB")
            except ImportError:
                pass
            
            end_time = time.time()
            processing_time = end_time - start_time
            formatted_time = format_time(processing_time)
            logger.info(f"Processing completed in {formatted_time}")

            # Move and rename output
            temp_vocal_path = temp_output_dir / input_name / "vocals.wav"
            if temp_vocal_path.exists():
                logger.info(f"Found vocals file: {temp_vocal_path}")
                os.rename(temp_vocal_path, output_file)
                logger.info(f"Renamed to: {output_file}")
                
                # Get output file size
                output_size_mb = get_file_size_mb(output_file)
                logger.info(f"Output file size: {output_size_mb:.2f} MB")
                
                print(f"Success! Isolated vocals saved as '{output_file}'")
                print(f"Processing time: {formatted_time}")
                print(f"Input file: {file_size_mb:.2f} MB, Output file: {output_size_mb:.2f} MB")
            else:
                logger.error(f"Vocal file not found at expected path: {temp_vocal_path}")
                print("Error: Vocal isolation failed. Check input file quality.")
                print("See log file for details: /audio/voice_isolation_cpu.log")

            # Clean up temporary files
            logger.info("Cleaning up temporary files")
            try:
                if temp_output_dir.exists():
                    for item in temp_output_dir.iterdir():
                        if item.is_dir():
                            for sub_item in item.iterdir():
                                sub_item.unlink(missing_ok=True)
                            item.rmdir()
                        else:
                            item.unlink(missing_ok=True)
                    temp_output_dir.rmdir()
                logger.info("Temporary files cleaned up successfully")
            except Exception as cleanup_error:
                logger.error(f"Error during cleanup: {str(cleanup_error)}")
                print(f"Warning: Could not clean up all temporary files: {str(cleanup_error)}")

        except KeyboardInterrupt:
            logger.warning("Process interrupted by user")
            print("\nProcess interrupted. Cleaning up...")
            try:
                if 'temp_output_dir' in locals() and temp_output_dir.exists():
                    import shutil
                    shutil.rmtree(temp_output_dir, ignore_errors=True)
            except Exception:
                pass
            print("Interrupted. You can try again with another file.")
            
        except Exception as e:
            logger.error(f"Error during processing: {str(e)}")
            logger.error(traceback.format_exc())
            print(f"Error during processing: {str(e)}")
            print("For more details, check the log file: /audio/voice_isolation_cpu.log")
            print("Try again with a different file or check file path.")

        # Ask to process another file
        again = input("\nProcess another file? (y/n): ").strip().lower()
        if again != 'y':
            logger.info("User finished processing. Exiting.")
            print("Exiting the tool. Goodbye!")
            break

if __name__ == "__main__":
    try:
        # Add psutil for memory tracking if not available
        try:
            import psutil
        except ImportError:
            logger.warning("psutil not installed. Memory tracking will be disabled.")
            print("Note: For better memory tracking, consider adding psutil to the container.")
        
        main()
    except Exception as e:
        logger.critical(f"Unhandled exception: {str(e)}")
        logger.critical(traceback.format_exc())
        print(f"Critical error: {str(e)}")
        print("Check the log file for details: /audio/voice_isolation_cpu.log")
        sys.exit(1)