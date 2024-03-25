import torch
import torch.nn as nn
from collections import namedtuple
from .components import Embeddings, Encoder




class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        
        self.device = config.device
        self.pad_id = config.pad_id

        self.encoder = Encoder(config)
        self.dropout = nn.Dropout(config.dropout_ratio)
        self.fc_out = nn.Linear(config.hidden_dim, 1)

        self.out = namedtuple('Out', 'logit loss')
        self.criterion = nn.BCEWithLogitsLoss()



    def forward(self, x, y=None):
        x_mask = (x == self.pad_id)
        x = self.encoder(x, x_mask)[:, 0, :]
        logit = self.fc_out(x).squeeze()

        if y is None:
            return logit

        self.out.logit = logit
        self.out.loss = self.criterion(logit, y)

        return self.out