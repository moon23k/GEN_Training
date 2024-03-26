import torch, os
import torch.nn as nn
from model import Generator, Discriminator




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



def load_generator(config):
    mode = config.mode

    model = Generator(config)
    init_weights(model)
    print("Initialized Generator has loaded")

    if mode == 'std_train':
        print_model_desc(model)
        return model.to(config.device)
    elif mode == 'gen_train':
        ckpt = 'ckpt/std_model.pt'
    elif mode == 'gan_train':
        ckpt = 'ckpt/gen_model.pt'
    elif mode != 'train':
        ckpt = f"ckpt/{config.strategy}_model.pt"
        
    model_state = torch.load(ckpt, map_location=config.device)['model_state_dict']    
    model.load_state_dict(model_state)
    print(f"Model states has loaded from {ckpt}")
    
    print_model_desc(model)
    return model.to(config.device)



def load_discriminator(config):
    model = Discriminator(config)
    init_weights(model)
    print("Initialized Discriminator model has loaded")


    ckpt = 'ckpt/discriminator.pt'
    if os.path.exists(ckpt):
        model_state = torch.load(ckpt, map_location=config.device)['model_state_dict']        
        model.load_state_dict(model_state)
        print(f"Model states has loaded from {ckpt}")                   

    print_model_desc(model)
    return model.to(config.device)    