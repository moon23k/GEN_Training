import json, torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence



class Dataset(torch.utils.data.Dataset):
    def __init__(self, mode, split):
        super().__init__()
        self.split = split
        self.data = self.load_data(split)

    @staticmethod
    def load_data(split):
        with open(f"data/{split}.json", 'r') as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if 'sample' in self.split:
            text = self.data[idx]['input_ids']
            summ = self.data[idx]['labels']            
            pred = self.data[idx]['pred']
            return ids, label
        else:
            text = self.data[idx]['input_ids']
            summ = self.data[idx]['labels']            
            return text, summ


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
    return DataLoader(Dataset(config.mode, split), 
                      batch_size=config.batch_size, 
                      shuffle=True if config.mode == 'train' else False, 
                      collate_fn=Collator(config.pad_id), 
                      num_workers=2)
        