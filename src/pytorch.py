import torch
import numpy as np

if torch.cuda.is_available():
    print("GPU is running")
else:
    print("No GPU")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
device_name = torch.cuda.get_device_name()
print("device", device, device_name)
# if __main__():
