import torch
print("CUDA disponible:", torch.cuda.is_available())
print("Versión:", torch.__version__)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
