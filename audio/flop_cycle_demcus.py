import time
import os
import sys
import psutil
import numpy as np
import torch
from demucs.pretrained import get_model
from demucs.apply import apply_model
from demucs.audio import AudioFile

def estimate_flops_per_cycle(audio_path):
    # Load model
    model = get_model("htdemucs")
    model.eval()
    
    # Load audio
    audio = AudioFile(audio_path).read(streams=0, samplerate=model.samplerate, channels=model.audio_channels)
    audio = audio.unsqueeze(0)
    segment = audio[:, :, :model.samplerate*10]  # 10 second segment
    
    # Warm up
    print("Warming up...")
    with torch.no_grad():
        _ = apply_model(model, segment)
    
    # Measure performance
    print("Measuring performance...")
    process = psutil.Process(os.getpid())
    start_cpu = process.cpu_times()
    start_time = time.time()
    
    with torch.no_grad():
        _ = apply_model(model, segment)
    
    end_time = time.time()
    end_cpu = process.cpu_times()
    
    # Calculate metrics
    wall_time = end_time - start_time
    cpu_time = (end_cpu.user - start_cpu.user) + (end_cpu.system - start_cpu.system)
    
    # Estimate FLOPS (Demucs has ~400M parameters)
    params = 400000000
    estimated_flops = params * 2
    
    # Estimate cycles (Apple Silicon ~3.2GHz)
    cpu_freq = 3200000000
    estimated_cycles = cpu_time * cpu_freq
    flops_per_cycle = estimated_flops / estimated_cycles if estimated_cycles > 0 else 0
    
    print(f"\nPerformance Metrics:")
    print(f"Wall Time: {wall_time:.2f}s")
    print(f"CPU Time: {cpu_time:.2f}s")
    print(f"Estimated FLOPS: {estimated_flops:,}")
    print(f"Estimated Cycles: {estimated_cycles:,.0f}")
    print(f"FLOPS/Cycle: {flops_per_cycle:.4f}")
    
    if flops_per_cycle > 16:
        print("\nRecommendation: Use GPU (AWS G4dn/G5 instances)")
    elif flops_per_cycle > 4:
        print("\nRecommendation: Both CPU/GPU viable")
    else:
        print("\nRecommendation: CPU processing is efficient")
    
    return flops_per_cycle

if __name__ == "__main__":
    audio_path = sys.argv[1] if len(sys.argv) > 1 else input("Enter audio path: ")
    estimate_flops_per_cycle(audio_path)