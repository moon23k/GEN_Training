import os, yaml, argparse, torch

from tokenizers import Tokenizer
from tokenizers.processors import TemplateProcessing

from module import (
    load_dataloader,
    load_generator,
    load_discriminator,
    Trainer,
    Tester,
    Generator
)



def set_seed(SEED=42):
    import random
    import numpy as np
    import torch.backends.cudnn as cudnn

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    cudnn.benchmark = False
    cudnn.deterministic = True



class Config(object):
    def __init__(self, args):    

        with open('config.yaml', 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
            for group in params.keys():
                for key, val in params[group].items():
                    setattr(self, key, val)

        self.mode = args.mode
        self.strategy = args.strategy
        self.search_method = args.search
        self.tokenizer_path = f'data/tokenizer.json'


        if self.mode in ['gen_train', 'gan_train']:
            self.lr *= 0.5


        if 'train' in self.mode:
            self.ckpt = f"ckpt/{self.mode[:3]}_model.pt"
        else:
            self.ckpt = f"ckpt/{self.strategy}_model.pt"
        

        use_cuda = torch.cuda.is_available()
        self.device_type = 'cuda' \
                           if use_cuda and self.mode != 'inference' \
                           else 'cpu'
        self.device = torch.device(self.device_type)


    def print_attr(self):
        for attribute, value in self.__dict__.items():
            print(f"* {attribute}: {value}")




def load_tokenizer(config):
    assert os.path.exists(config.tokenizer_path)

    tokenizer = Tokenizer.from_file(config.tokenizer_path)
    tokenizer.post_processor = TemplateProcessing(
        single=f"{config.bos_token} $A {config.eos_token}",
        special_tokens=[(config.bos_token, config.bos_id), 
                        (config.eos_token, config.eos_id)]
        )
    
    return tokenizer




def main(args):
    
    set_seed()

    config = Config(args)
    tokenizer = load_tokenizer(config)

    generator = load_generator(config)
    discriminator = load_discriminator(config) if config.mode == 'gan_train' else None
    

    #For Training Process
    if 'train' in config.mode:
        if config.mode == 'gan_train':
            sampler = Sampler(config, generator, tokenizer)
            sampler.sample()            

        train_dataloader = load_dataloader(config, tokenizer, 'train')
        valid_dataloader = load_dataloader(config, tokenizer, 'valid')
        
        trainer_kwargs = {
            'generator': generator,
            'discriminator': discriminator,
            'train_dataloader': train_dataloader,
            'valid_dataloader': valid_dataloader
        }

        trainer = Trainer(config, trainer_kwargs)
        trainer.train()

    
    #For Testing Process
    elif config.mode == 'test':
        test_dataloader = load_dataloader(config, tokenizer, 'test')
        tester = Tester(config, generator, tokenizer, test_dataloader)
        tester.test()
    

    #For Inference Process
    elif config.mode == 'inference':
        generator = Generator(config, generator, tokenizer)
        generator.inference()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', required=True)
    parser.add_argument('-strategy', default='std', required=False)
    parser.add_argument('-search', default='greedy', required=False)
    
    args = parser.parse_args()
    assert args.mode.lower() in ['std_train', 'gen_train', 'gan_train', 'test', 'inference']
    assert args.strategy.lower() in ['std', 'gen', 'gan']
    assert args.search.lower() in ['greedy', 'beam']

    if 'train' in args.mode:
        if args.mode == 'gen_train':
            assert os.path.exists(f'ckpt/std_model.pt')
        elif args.mode == 'gan_train':
            assert os.path.exists(f'ckpt/gen_model.pt')
    else:
        if args.strategy == 'std':
            assert os.path.exists(f'ckpt/std_model.pt')
        elif args.strategy == 'gen':
            assert os.path.exists(f'ckpt/gen_model.pt')
        elif args.strategy == 'gan':
            assert os.path.exists(f'ckpt/gan_model.pt')            
    main(args)