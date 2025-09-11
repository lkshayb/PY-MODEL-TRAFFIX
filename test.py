#File to test CUDA is working or not.
import torch
print("RUNNING ON TORCH VERSION ===> ",torch.__version__)                
print("RUNNING ON CUDA VERSION ===>",torch.version.cuda)                
print("CUDA AVAILABILITY ===> ",torch.cuda.is_available())          
if(torch.cuda.is_available()):
    print("GPU DEVICE NAME ===>",torch.cuda.get_device_name(0))     
