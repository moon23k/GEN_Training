import json, torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence



class Dataset(torch.utils.data.Dataset):
    def __init__(self, split):
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
        
        if 'sample' not in self.split:
            text = self.data[idx]['text']
            summ = self.data[idx]['summ']            
            return text, summ
        
        else:
            text = self.data[idx]['text']
            summ = self.data[idx]['summ']            
            pred = self.data[idx]['pred']
            return text, summ, pred


class Collator(object):
    def __init__(self, split, pad_id):
        self.split = split
        self.pad_id = pad_id

    def __call__(self, batch):
        
        if 'sample' not in self.split:
            text_batch, summ_batch = [], []
            for elem in batch:
                text_batch.append(torch.LongTensor(elem[0])) 
                summ_batch.append(torch.LongTensor(elem[1]))

            return {'text': self.pad_batch(text_batch),
                    'summ': self.pad_batch(summ_batch)}

        else:
            text_batch, summ_batch, pred_batch = [], [], []
            for elem in batch:
                text_batch.append(torch.LongTensor(elem[0])) 
                summ_batch.append(torch.LongTensor(elem[1]))
                pred_batch.append(torch.LongTensor(elem[2]))

            return {'text': self.pad_batch(text_batch),
                    'summ': self.pad_batch(summ_batch),
                    'pred': self.pad_batch(pred_batch)}

    def pad_batch(self, batch):
        return pad_sequence(batch, batch_first=True, padding_value=self.pad_id)


def load_dataloader(config, split):
    return DataLoader(Dataset(split), 
                      batch_size=config.batch_size, 
                      shuffle=True if config.mode == 'train' else False, 
                      collate_fn=Collator(split, config.pad_id), 
                      num_workers=2)