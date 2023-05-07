import os, json, argparse, torch

from module.test import Tester
from module.train import Trainer
from module.data import load_dataloader
from module.pretrain import GenTrainer, DisTrainer
from module.model import load_generator, load_discriminator

from tqdm import tqdm
from transformers import (set_seed, 
                          LongformerModel, 
                          LEDTokenizerFast,
                          LEDForConditionalGeneration)



class Config(object):
    def __init__(self, mode):    
        self.mode = mode
        self.g_mname = "allenai/led-base-16384"
        self.d_mname = "allenai/longformer-base-4096"
        self.max_len = 4096

        self.clip = 1
        self.lr = 5e-5
        self.n_epochs = 10
        self.batch_size = 32
        self.iters_to_accumulate = 4

        self.early_stop = True
        self.patience = 3

        if self.mode == 'inference':
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.g_ckpt = 'ckpt/generator.pt'
        self.d_ckpt = 'ckpt/discriminator.pt'
        self.g_base_ckpt = 'ckpt/generator_base.pt'
        self.d_base_ckpt = 'ckpt/discriminator_base.pt'


    def print_attr(self):
        for attribute, value in self.__dict__.items():
            print(f"* {attribute}: {value}")



def generate(model, dataloader, split):
    generated = []
    for batch in tqdm(dataloader):
        text, summ = batch[0], batch[1]
        batch_size = len(text)

        text_encodings = tokenizer(text, padding=True, truncation=True, 
                                   return_tensors='pt').to(model.device)        
        
        with torch.no_grad():
            pred = model.generate(input_ids=text_encodings.input_ids,
                                  attention_mask=text_encodings.attention_mask,
                                  max_new_tokens=1024, use_cache=True)

        for i in range(batch_size):
            generated.append({'text': text[i], 
                              'summ': summ[i], 
                              'pred': pred[i]})

    with open(f'data/{split}.json', 'w') as f:
        json.dump(generated, f)        
    



def pretrain(config, g_model, d_model, tokenizer):

    ###PreTrain Generator with Character Dataset    
    g_train_dataloader = load_dataloader(config, 'train')
    g_valid_dataloader = load_dataloader(config, 'valid')

    g_trainer = GenTrainer(config, g_model, tokenizer, 
                           g_train_dataloader, g_valid_dataloader)
    g_trainer.train()

    #Load Best Generator States After Pretrtaining
    g_model_state = torch.torch.load(config.g_base_ckpt, map_location=config.device)['model_state_dict']
    g_model.load_state_dict(model_state)
    g_model.eval()

    ###Generate Samples for Discriminator PreTraining
    generate(config, g_model, g_train_dataloader)
    generate(config, g_model, g_valid_dataloader)

    
    ###PreTrain Discriminator
    config.model_type = 'discriminator'
    d_train_dataloader = load_dataloader(config, 'sample_train')
    d_valid_dataloader = load_dataloader(config, 'sample_valid')        

    d_trainer = DisTrainer(config, d_model, tokenizer, 
                           d_train_dataloader, d_valid_dataloader)
    d_trainer.train()




def train(config, g_model, d_model, tokenizer):
    train_dataloader = load_dataloader(config, 'train')
    valid_dataloader = load_dataloader(config, 'valid')

    trainer = Trainer(config, g_model, d_model, tokenizer, 
                      train_dataloader, valid_dataloader)
    trainer.train()



def test(config, g_model, d_model, tokenizer):
    test_dataloader = load_dataloader(config, 'test')
    tester = Tester(config, g_model, d_model, tokenizer, test_dataloader)    
    tester.test()    



def inference(g_model, tokenizer):
    g_model.eval()
    print(f'--- Inference Process Started! ---')
    print('[ Type "quit" on user input to stop the Process ]')
    
    while True:
        input_seq = input('\nUser Input Sequence >> ').lower()

        #End Condition
        if input_seq == 'quit':
            print('\n--- Inference Process has terminated! ---')
            break        

        #convert user input_seq into model input_ids
        input_ids = tokenizer(input_seq, return_tensors='pt')['input_ids']
        output_ids = g_model.generate(input_ids, max_new_tokens=1024, use_cache=True)
        output_seq = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

        #Search Output Sequence
        print(f"Model Out Sequence >> {output_seq}")



def main(args):
    set_seed(42)
    config = Config(args)    
    tokenizer = LEDTokenizerFast.from_pretrained(config.g_mname)
    tokenizer.model_max_length = config.max_len
    setattr(config, 'pad_id', tokenizer.pad_token_id)

    g_model = load_generator(config)
    if config.mode != "inference":
        d_model = load_discriminator(config)

    if config.mode == 'pretrain':
        pretrain(config, g_model, d_model, tokenizer)
    elif config.mode == 'train':
        train(config, g_model, d_model, tokenizer)
    elif config.mode == 'test':
        test(config, g_model, d_model, tokenizer)
    elif config.mode == 'inference':
        inference(g_model, tokenizer)
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', required=True)
    
    args = parser.parse_args()
    assert args.mode.lower() in ['pretrain', 'train', 'test', 'inference']

    if args.mode == 'train':
        assert os.path.exists('ckpt/generator_base.pt')
        assert os.path.exists('ckpt/discriminator_base.pt')
    
    elif args.mode in ['test', 'inference']:
        assert os.path.exists('ckpt/discriminator.pt')
        assert os.path.exists('ckpt/generator.pt')

    main(args.mode)