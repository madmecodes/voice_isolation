import os
import sys
import time
import logging
import traceback
import subprocess
from pathlib import Path
from spleeter.separator import Separator
import numpy as np
import librosa
import soundfile as sf

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/audio/batch_voice_isolation.log')
    ]
)
logger = logging.getLogger(__name__)

# Constants
BATCH_SIZE_SECONDS = 600  # 10 minutes per batch
OVERLAP_SECONDS = 5  # 5 second overlap between batches to avoid artifacts at boundaries
SAMPLE_RATE = 44100  # Standard sample rate

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

def get_audio_duration(file_path):
    """Get the duration of an audio file in seconds."""
    try:
        duration = librosa.get_duration(path=file_path)
        return duration
    except Exception as e:
        logger.error(f"Error getting audio duration: {str(e)}")
        # Fallback using ffprobe
        try:
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 
                 'default=noprint_wrappers=1:nokey=1', file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            return float(result.stdout.strip())
        except Exception as ffprobe_error:
            logger.error(f"Error using ffprobe fallback: {str(ffprobe_error)}")
            return -1

def split_audio(file_path, output_dir, batch_size=BATCH_SIZE_SECONDS, overlap=OVERLAP_SECONDS):
    """Split audio file into batches with overlap."""
    logger.info(f"Loading audio file: {file_path}")
    print("Loading audio file...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get audio duration to calculate number of batches
    duration = get_audio_duration(file_path)
    if duration <= 0:
        logger.error("Failed to determine audio duration")
        return []
    
    # Calculate number of batches
    if duration <= batch_size:
        # File is small enough, no need to split
        logger.info(f"Audio duration ({duration:.2f}s) is less than batch size, no splitting needed")
        # Create a symbolic link or copy to avoid loading the file twice
        batch_path = os.path.join(output_dir, "batch_full.wav")
        if file_path.lower().endswith('.wav'):
            # Create symlink for WAV files
            if os.path.exists(batch_path):
                os.remove(batch_path)
            os.symlink(os.path.abspath(file_path), batch_path)
        else:
            # Convert to WAV for non-WAV files
            logger.info(f"Converting {file_path} to WAV format")
            subprocess.run([
                'ffmpeg', '-i', file_path, '-ar', str(SAMPLE_RATE), batch_path
            ], check=True)
        return [batch_path]
    
    # Longer file, need to split into batches
    num_batches = int(np.ceil((duration - overlap) / (batch_size - overlap)))
    logger.info(f"Audio duration: {duration:.2f}s, splitting into {num_batches} batches")
    print(f"Audio duration: {format_time(duration)}, splitting into {num_batches} batches...")
    
    batch_files = []
    for i in range(num_batches):
        # Calculate start and end times for this batch
        start_time = max(0, i * (batch_size - overlap))
        end_time = min(duration, start_time + batch_size)
        
        batch_path = os.path.join(output_dir, f"batch_{i:03d}.wav")
        logger.info(f"Creating batch {i+1}/{num_batches}: {start_time:.2f}s to {end_time:.2f}s")
        
        # Use ffmpeg to extract the segment
        subprocess.run([
            'ffmpeg', '-y', '-i', file_path, 
            '-ss', str(start_time), '-to', str(end_time),
            '-ar', str(SAMPLE_RATE), batch_path
        ], check=True)
        
        batch_files.append(batch_path)
        print(f"Prepared batch {i+1}/{num_batches} ({start_time:.1f}s to {end_time:.1f}s)")
    
    return batch_files

def process_batch(batch_file, output_dir, separator):
    """Process a single batch using Spleeter."""
    logger.info(f"Processing batch: {batch_file}")
    
    # Get the base filename without extension
    batch_name = os.path.basename(batch_file)
    batch_stem = os.path.splitext(batch_name)[0]
    
    # Create temporary output directory for this batch
    batch_output_dir = os.path.join(output_dir, f"{batch_stem}_temp")
    os.makedirs(batch_output_dir, exist_ok=True)
    
    # Process batch with spleeter
    logger.info(f"Running spleeter on {batch_file}")
    try:
        separator.separate_to_file(batch_file, batch_output_dir)
        
        # Get path to the vocals output
        vocals_path = os.path.join(batch_output_dir, batch_stem, "vocals.wav")
        if not os.path.exists(vocals_path):
            logger.error(f"Vocals file not found: {vocals_path}")
            return None
        
        return vocals_path
    except Exception as e:
        logger.error(f"Error processing batch {batch_file}: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def merge_audio_files(vocal_files, output_file, overlap=OVERLAP_SECONDS):
    """Merge multiple vocal files with crossfade between them."""
    logger.info(f"Merging {len(vocal_files)} vocal files with {overlap}s overlap")
    print(f"Merging {len(vocal_files)} processed segments...")
    
    if len(vocal_files) == 1:
        # Just copy the single file
        logger.info("Only one file to merge, copying directly")
        subprocess.run(['cp', vocal_files[0], output_file], check=True)
        return True
    
    # Use ffmpeg to concatenate with crossfade
    concat_file = os.path.splitext(output_file)[0] + "_concat_list.txt"
    with open(concat_file, "w") as f:
        for vfile in vocal_files:
            f.write(f"file '{os.path.abspath(vfile)}'\n")
    
    # Use ffmpeg filter complex to concatenate with crossfade
    filter_complex = ""
    for i in range(len(vocal_files) - 1):
        if i == 0:
            filter_complex += f"[0:0][1:0]acrossfade=d={overlap}:c1=tri:c2=tri[out1];"
        else:
            filter_complex += f"[out{i}][{i+1}:0]acrossfade=d={overlap}:c1=tri:c2=tri[out{i+1}];"
    
    # Construct the ffmpeg command
    ffmpeg_cmd = [
        'ffmpeg', '-y', '-f', 'concat', '-safe', '0', 
        '-i', concat_file, '-filter_complex', filter_complex
    ]
    
    # Add the output name correctly
    last_output = f"[out{len(vocal_files)-1}]"
    ffmpeg_cmd += ['-map', last_output, output_file]
    
    try:
        # Run the command
        subprocess.run(ffmpeg_cmd, check=True)
        
        # Clean up the concat file
        os.remove(concat_file)
        
        logger.info(f"Successfully merged files to {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error merging audio files: {str(e)}")
        
        # Fallback to simpler method if complex filter fails
        logger.info("Trying simpler concatenation method...")
        try:
            # Simple concatenation without crossfade
            subprocess.run([
                'ffmpeg', '-y', '-f', 'concat', '-safe', '0', 
                '-i', concat_file, '-c', 'copy', output_file
            ], check=True)
            
            os.remove(concat_file)
            logger.info(f"Successfully merged files using simple method to {output_file}")
            return True
        except subprocess.CalledProcessError as e2:
            logger.error(f"Error in fallback merging: {str(e2)}")
            return False

def cleanup_temp_files(temp_dir):
    """Clean up temporary files and directories."""
    logger.info(f"Cleaning up temporary directory: {temp_dir}")
    try:
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        logger.info("Cleanup completed")
        return True
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")
        return False

def main():
    logger.info("=== Batch Voice Isolation Tool (Powered by Spleeter) ===")
    print("=== Batch Voice Isolation Tool (Powered by Spleeter) ===")
    print("This script processes audio files of any length by splitting into batches.")
    print("GPU version will be used automatically if available.")
    logger.info("Tool started")
    
    # Check for GPU
    try:
        import tensorflow as tf
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            try:
                for device in physical_devices:
                    tf.config.experimental.set_memory_growth(device, True)
                print(f"GPU acceleration enabled. Found {len(physical_devices)} GPU(s).")
                logger.info(f"GPU acceleration enabled. Found {len(physical_devices)} GPU(s)")
            except Exception as e:
                print(f"Error configuring GPU: {str(e)}")
                logger.error(f"Error configuring GPU: {str(e)}")
        else:
            print("No GPU found. Running on CPU. Processing may be slower.")
            logger.warning("No GPU found. Running on CPU")
    except Exception as tf_error:
        logger.error(f"Error checking for GPU: {str(tf_error)}")
        print("Unable to check for GPU. Continuing with CPU processing.")
    
    print("\nInstructions:")
    print("- Enter the full path to your audio file (e.g., /audio/interview.mp3).")
    print("- Make sure your audio files are in the 'audio' directory that's mounted to the container.")
    print("- Output will be saved as '<filename>_isolated.wav' in the same directory.")
    print("- For files longer than 10 minutes, the audio will be processed in batches.")
    print("- Logs are saved to /audio/batch_voice_isolation.log")
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
            input_path_obj = Path(input_path)
            input_dir = input_path_obj.parent
            input_name = input_path_obj.stem
            output_file = input_dir / f"{input_name}_isolated.wav"

            # Create a temporary output directory
            temp_output_dir = input_dir / f"{input_name}_batch_temp"
            os.makedirs(temp_output_dir, exist_ok=True)
            logger.info(f"Created temporary directory: {temp_output_dir}")

            # Get audio duration
            duration = get_audio_duration(input_path)
            
            # Start timing for whole process
            start_time_total = time.time()
            
            # Print info
            print(f"\nProcessing '{input_path_obj.name}' to isolate vocals...")
            print(f"File size: {file_size_mb:.2f} MB, Duration: {format_time(duration)}")
            
            # Determine if batching is needed
            if duration > BATCH_SIZE_SECONDS:
                print(f"Audio is longer than {BATCH_SIZE_SECONDS/60:.1f} minutes, processing in batches...")
                logger.info(f"Audio duration ({duration:.2f}s) exceeds batch size, using batch processing")
            else:
                print("Audio is short enough to process in one go.")
                logger.info(f"Audio duration ({duration:.2f}s) is within batch size, processing as single batch")
            
            # Split audio into batches
            batch_dir = temp_output_dir / "batches"
            batch_files = split_audio(input_path, batch_dir)
            
            if not batch_files:
                raise Exception("Failed to split audio into batches")
            
            # Process each batch
            vocal_files = []
            for i, batch_file in enumerate(batch_files):
                print(f"\nProcessing batch {i+1}/{len(batch_files)}...")
                logger.info(f"Processing batch {i+1}/{len(batch_files)}: {batch_file}")
                
                start_time_batch = time.time()
                vocal_file = process_batch(batch_file, temp_output_dir, separator)
                
                if vocal_file:
                    vocal_files.append(vocal_file)
                    end_time_batch = time.time()
                    batch_time = end_time_batch - start_time_batch
                    print(f"Batch {i+1} processed in {format_time(batch_time)}")
                else:
                    logger.error(f"Failed to process batch {i+1}")
                    print(f"Failed to process batch {i+1}. See logs for details.")
            
            # Check if at least one batch was processed successfully
            if not vocal_files:
                raise Exception("All batches failed to process")
            
            # Merge processed batches
            print("\nMerging processed audio segments...")
            merge_success = merge_audio_files(vocal_files, output_file)
            
            if not merge_success:
                raise Exception("Failed to merge processed audio segments")
            
            # Calculate total processing time
            end_time_total = time.time()
            total_time = end_time_total - start_time_total
            formatted_time = format_time(total_time)
            
            # Get output file size
            output_size_mb = get_file_size_mb(output_file)
            
            # Final success message
            print(f"\nSuccess! Isolated vocals saved as '{output_file}'")
            print(f"Processing time: {formatted_time}")
            print(f"Input file: {file_size_mb:.2f} MB, Output file: {output_size_mb:.2f} MB")
            
            # Log success
            logger.info(f"Processing completed successfully in {formatted_time}")
            logger.info(f"Output file: {output_file} (Size: {output_size_mb:.2f} MB)")
            
            # Clean up temporary files
            print("Cleaning up temporary files...")
            cleanup_temp_files(temp_output_dir)

        except KeyboardInterrupt:
            logger.warning("Process interrupted by user")
            print("\nProcess interrupted. Cleaning up...")
            try:
                if 'temp_output_dir' in locals() and os.path.exists(temp_output_dir):
                    cleanup_temp_files(temp_output_dir)
            except Exception:
                pass
            print("Interrupted. You can try again with another file.")
            
        except Exception as e:
            logger.error(f"Error during processing: {str(e)}")
            logger.error(traceback.format_exc())
            print(f"Error during processing: {str(e)}")
            print("For more details, check the log file: /audio/batch_voice_isolation.log")
            print("Try again with a different file or check file path.")

        # Ask to process another file
        again = input("\nProcess another file? (y/n): ").strip().lower()
        if again != 'y':
            logger.info("User finished processing. Exiting.")
            print("Exiting the tool. Goodbye!")
            break

if __name__ == "__main__":
    try:
        # Additional dependencies needed for batch processing
        print("Checking for required dependencies...")
        required_packages = [
            ('librosa', 'librosa'),
            ('soundfile', 'soundfile'),
            ('numpy', 'numpy')
        ]
        
        missing_packages = []
        for package_name, import_name in required_packages:
            try:
                __import__(import_name)
            except ImportError:
                missing_packages.append(package_name)
        
        if missing_packages:
            print(f"Installing missing dependencies: {', '.join(missing_packages)}")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)
            print("Dependencies installed successfully")
        
        main()
    except Exception as e:
        logger.critical(f"Unhandled exception: {str(e)}")
        logger.critical(traceback.format_exc())
        print(f"Critical error: {str(e)}")
        print("Check the log file for details: /audio/batch_voice_isolation.log")
        sys.exit(1)