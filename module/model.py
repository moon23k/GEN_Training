import torch, os
import torch.nn as nn
from model import Transformer




def init_weights(model):
    for name, param in model.named_parameters():
        if 'weight' in name and 'norm' not in name:
            nn.init.xavier_uniform_(param)            



def print_model_desc(model):
    #Number of trainerable parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"--- Model Params: {n_params:,}")

    #Model size check
    param_size, buffer_size = 0, 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print(f"--- Model  Size : {size_all_mb:.3f} MB\n")



def load_model(config):
    mode = config.mode
    model_type = config.model_type

    model = Transformer(config)
    init_weights(model)
    print(f"Initialized {model_type} model has loaded")

    if mode != 'train' or model_type == 'generative':
        if model_type == 'generative' and mode == 'train':
            ckpt = f'ckpt/{config.task}/baseline_model.pt'        
        else:
            ckpt = config.ckpt
        assert os.path.exists(ckpt)
        
        model_state = torch.load(ckpt, map_location=config.device)['model_state_dict']
        
        model.load_state_dict(model_state)
        print(f"Model states has loaded from {ckpt}")       
    
    print_model_desc(model)
    return model.to(config.device)