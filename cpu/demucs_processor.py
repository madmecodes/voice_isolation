import os
import sys
import time
import torch
import numpy as np
import soundfile as sf
import psutil
from pathlib import Path
from demucs.pretrained import get_model
from demucs.apply import apply_model
from demucs.audio import AudioFile

def process_audio(audio_path):
    # Load model
    print("Initializing Demucs...")
    model = get_model("htdemucs")
    model.eval()
    
    # Get file details
    input_path = Path(audio_path)
    input_dir = input_path.parent
    input_name = input_path.stem
    file_size_mb = os.path.getsize(input_path) / (1024 * 1024)
    
    # Load audio
    print(f"Loading audio file: {input_path.name} ({file_size_mb:.2f} MB)")
    audio = AudioFile(audio_path).read(streams=0, samplerate=model.samplerate, channels=model.audio_channels)
    audio = audio.unsqueeze(0)
    
    # Start performance measurement
    process = psutil.Process(os.getpid())
    start_cpu = process.cpu_times()
    start_mem = process.memory_info().rss / (1024 * 1024)
    start_time = time.time()
    
    # Process audio
    print("Processing audio to isolate vocals...")
    with torch.no_grad():
        sources = apply_model(model, audio)
    
    # End performance measurement
    end_time = time.time()
    end_cpu = process.cpu_times()
    end_mem = process.memory_info().rss / (1024 * 1024)
    
    # Calculate metrics
    wall_time = end_time - start_time
    cpu_time = (end_cpu.user - start_cpu.user) + (end_cpu.system - start_cpu.system)
    mem_usage = end_mem - start_mem
    
    # FLOPS/cycle calculation
    params = 400000000
    estimated_flops = params * 2
    cpu_freq = 3200000000
    estimated_cycles = cpu_time * cpu_freq
    flops_per_cycle = estimated_flops / estimated_cycles if estimated_cycles > 0 else 0
    
    # Save audio - fixed to handle dimensions correctly
    sources_list = model.sources
    vocals_idx = sources_list.index("vocals")
    output_file = input_dir / f"{input_name}_demucs_vocals.wav"
    
    # Extract and prepare vocals for saving
    vocals = sources[:, vocals_idx].squeeze(0).cpu().numpy()
    sf.write(str(output_file), vocals.T, model.samplerate)
    output_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    
    # Format time
    def format_time(seconds):
        if seconds < 60:
            return f"{seconds:.1f}s"
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    
    # Print performance metrics
    print("\n=== Performance Metrics ===")
    print(f"Processing time: {format_time(wall_time)}")
    print(f"CPU time: {format_time(cpu_time)} (CPU usage: {cpu_time/wall_time*100:.1f}%)")
    print(f"Memory usage: {mem_usage:.1f} MB")
    print(f"FLOPS/cycle: {flops_per_cycle:.4f}")
    print(f"Input file: {file_size_mb:.2f} MB, Output file: {output_size_mb:.2f} MB")
    
    # Print recommendation
    if flops_per_cycle > 16:
        print("\nRecommendation: Use GPU (AWS G4dn/G5 instances)")
    elif flops_per_cycle > 4:
        print("\nRecommendation: Both CPU/GPU viable")
    else:
        print("\nRecommendation: CPU processing is efficient")
    
    print(f"\nSuccess! Isolated vocals saved as '{output_file}'")
    
    return {
        "wall_time": wall_time,
        "cpu_time": cpu_time,
        "mem_usage": mem_usage,
        "flops_per_cycle": flops_per_cycle,
        "output_file": output_file
    }

if __name__ == "__main__":
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
    else:
        audio_path = input("Enter audio path: ")
    
    if not os.path.exists(audio_path):
        print(f"Error: File '{audio_path}' not found")
        sys.exit(1)
    
    process_audio(audio_path)