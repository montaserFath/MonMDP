import torch
import numpy as np

if torch.cuda.is_available():
    print("GPU is running")
else:
    print("No GPU")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# if __main__():
