import json, torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence



class Dataset(torch.utils.data.Dataset):
    def __init__(self, split):
        super().__init__()
        self.data = self.load_data(split)

    @staticmethod
    def load_data(split):
        with open(f"data/{split}.json", 'r') as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        ids = self.data[idx]['input_ids']
        label = self.data[idx]['labels']
        
        return {'input_ids': ids, 'labels': label}



class Collator(object):
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        ids_batch, label_batch = [], []
        
        for elem in batch:
            ids_batch.append(torch.LongTensor(elem['input_ids'])) 
            label_batch.append(torch.LongTensor(elem['labels']))

        return {'input_ids': self.pad_batch(ids_batch),
                'labels': self.pad_batch(label_batch)}

    def pad_batch(self, batch):
        return pad_sequence(batch, batch_first=True, padding_value=self.pad_id)


def load_dataloader(config, split):
    return DataLoader(Dataset(split), 
                      batch_size=config.batch_size, 
                      shuffle=True if 'train' in config.mode else False, 
                      collate_fn=Collator(config.pad_id), 
                      num_workers=2)
        