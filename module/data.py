import json, torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence



class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, tokenizer, split):
        super().__init__()

        self.tokenizer = tokenizer
        self.discriminative = config.discriminative
        self.data = self.load_data(split)


    def load_data(self, split):
        if self.discriminative:
            split = f'sample_{split}'

        with open(f"data/{split}.json", 'r') as f:
            data = json.load(f)

        return data


    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, idx):
        x = self.tokenizer.encode(self.data[idx]['x']).ids
        y = [self.data[idx]['y']] if self.discriminative \
            else self.tokenizer.encode(self.data[idx]['y']).ids
        return torch.LongTensor(x), torch.LongTensor(y)



class Collator(object):
    def __init__(self, config):
        self.pad_id = config.pad_id
        self.discriminative = config.discriminative


    def __call__(self, batch):
        x_batch, y_batch = zip(*batch)
        
        return {
            'x': self.pad_batch(x_batch), 
            'y': torch.Tensor(y_batch) if self.discriminative \
                 else self.pad_batch(y_batch)
                }


    def pad_batch(self, batch):
        return pad_sequence(
            batch, 
            batch_first=True, 
            padding_value=self.pad_id
        )



def load_dataloader(config, tokenizer, split):
    return DataLoader(
        Dataset(config, tokenizer, split), 
        batch_size=config.batch_size, 
        shuffle=split == 'train',
        collate_fn=Collator(config),
        pin_memory=True,
        num_workers=2
    )