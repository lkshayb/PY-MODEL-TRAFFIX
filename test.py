import torch
print(torch.__version__)                  # should be >= 2.5.0
print(torch.version.cuda)                 # should NOT be None
print(torch.cuda.is_available())          # should be True
print(torch.cuda.get_device_name(0))      # should say RTX 4050
