import os
import torch


script_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
#models_path = os.path.join(script_path, "models")
models_path = './models/'
models_weights = {'HAT': 'HAT_SRx4_ImageNet-pretrain.pth', 
                  'HDSRNet': 'HDSRNet_x4.pt'}

no_half = True
cpu = torch.device("cpu")
device = 'cuda'
dtype = torch.float16
dtype_vae = torch.float16


outputpath = './outputs/'
MAX_DURATION_BATCH = 200