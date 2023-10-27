import json, torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence



class Dataset(torch.utils.data.Dataset):

    def __init__(self, config, tokenizer, split):
        super().__init__()
        self.model_type = config.model_type
        self.tokenizer = tokenizer
        self.data = self.load_data(config.task, split)


    def load_data(self, task, split):
        with open(f"data/{task}/{split}.json", 'r') as f:
            data = json.load(f)

        if self.model_type == 'augment':
            return data
        return data[::10]


    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, idx):
        x = self.tokenizer.encode(self.data[idx]['x']).ids
        y = self.tokenizer.encode(self.data[idx]['y']).ids
        return torch.LongTensor(x), torch.LongTensor(y)



class Collator(object):

    def __init__(self, pad_id):
        self.pad_id = pad_id


    def __call__(self, batch):
        x_batch, y_batch = zip(*batch)     

        return {'x': self.pad_batch(x_batch), 
                'y': self.pad_batch(y_batch)}


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
        collate_fn=Collator(config.pad_id),
        pin_memory=True,
        num_workers=2
    )