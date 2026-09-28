import os
import json
import torch

def setup_environment():
    dirs = ['data', 'checkpoints', 'runs', 'outputs']
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"Ensured directory exists: {d}/")

    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "No NVIDIA GPU found"
    
    env_info = {
        "cuda_available": cuda_available,
        "device_name": device_name,
        "pytorch_version": torch.__version__,
        "dirs_created": dirs
    }
    
    with open('environment.json', 'w') as f:
        json.dump(env_info, f, indent=4)
        
    print(f"Environment setup complete.")
    print(f"CUDA Available: {cuda_available}")
    print(f"Device Name: {device_name}")
    print(f"PyTorch Version: {torch.__version__}")

if __name__ == "__main__":
    setup_environment()
