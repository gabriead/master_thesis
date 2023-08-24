import torch

print(True if torch.cuda.is_available() else False)
print("Available GPUs:", tf.config.list_physical_devices("GPU"))