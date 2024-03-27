import torch
import torch.nn as nn
from .components import clones, Embeddings, Encoder





class DecoderLayer(nn.TransformerDecoderLayer):
    def forward(
        self,
        x,
        memory=None,
        e_mask=None,
        d_mask=None,
        use_cache=False
    ):

        if not use_cache:
            return super().forward(
                x,
                memory,
                memory_key_padding_mask=e_mask,
                tgt_mask=d_mask
            )


        last_token = x[:, -1:, :]

        # self attention part
        _x = self.self_attn(last_token, x, x)[0]

        last_token = last_token + self.dropout1(_x)
        last_token = self.norm1(last_token)


        # encoder-decoder attention
        _x = self.multihead_attn(
            last_token, memory, memory,
            key_padding_mask=e_mask,
        )[0]

        last_token = last_token + self.dropout2(_x)
        last_token = self.norm2(last_token)

        # final feed-forward network
        _x = self.activation(self.linear1(last_token))
        _x = self.linear2(self.dropout(_x))
        last_token = last_token + self.dropout3(_x)
        last_token = self.norm3(last_token)
        
        return last_token



class Decoder(nn.TransformerDecoder):

    def forward(
        self,
        x,
        memory=None,
        cache=None,
        e_mask=None,
        d_mask=None,
        use_cache=True
    ):

        output = x

        #In case of not using Cache
        if not use_cache:
            for layer in self.layers:
                output = layer(output, memory, e_mask, d_mask, False)
            return output, None

        #In case of using Cache
        new_token_cache = []
        for idx, layer in enumerate(self.layers):
            output = layer(output, memory, use_cache=True)
            new_token_cache.append(output)
            
            if cache is not None:  
                output = torch.cat([cache[idx], output], dim=1)

        new_cache = torch.stack(new_token_cache, dim=0)

        if cache is not None:
            new_cache = torch.cat([cache, new_cache], dim=2)

        return output, new_cache




class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()

        self.bos_id = config.bos_id
        self.eos_id = config.eos_id
        self.pad_id = config.pad_id

        self.mode = config.mode
        self.device = config.device
        self.max_len = config.max_len
        self.vocab_size = config.vocab_size

        self.enc_emb = Embeddings(config)
        self.encoder = Encoder(config)

        self.dec_emb = Embeddings(config)
        self.decoder = Decoder(
            DecoderLayer(
                d_model=config.hidden_dim, 
                nhead=config.n_heads, 
                dim_feedforward=config.pff_dim,
                batch_first=True
            ),
            num_layers=config.n_layers,
        )

        self.generator = nn.Linear(config.hidden_dim, self.vocab_size)


    @staticmethod
    def shift_y(y):
        return y[:, :-1], y[:, 1:]


    def pad_mask(self, x):
        return x == self.pad_id


    def dec_mask(self, x):
        sz = x.size(1)
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1).to(self.device)


    def encode(self, x, x_mask):
        x = self.enc_emb(x)
        x = self.encoder(x, x_mask)
        return x


    def decode(
        self, x, memory, cache=None, 
        e_mask=None, d_mask=None, use_cache=False
        ):
        
        x = self.dec_emb(x)
        x, cache = self.decoder(x, memory, cache, e_mask, d_mask, use_cache)
        return x, cache        
        

    def standard_forward(self, x, y):        
        e_mask = self.pad_mask(x)
        d_mask = self.dec_mask(y)

        memory = self.encode(x, e_mask)

        dec_out, _ = self.decode(y, memory, None, e_mask, d_mask, use_cache=False)
        logit = self.generator(dec_out)

        return logit
    

    def generative_forward(self, x, y):
        batch_size, output_len = y.shape
        logit = torch.empty(batch_size, output_len, self.vocab_size).to(self.device)

        pred = torch.zeros((batch_size, 1), dtype=torch.long)
        pred = pred.fill_(self.bos_id).to(self.device)

        cache=None
        e_mask = self.pad_mask(x)
        memory = self.encode(x, e_mask)

        for idx in range(1, output_len+1):
            y = pred[:, :idx]
            d_out, cache = self.decode(y, memory, cache, e_mask, use_cache=True)

            curr_logit = self.generator(d_out[:, -1:, :])
            curr_pred = curr_logit.argmax(dim=-1)

            logit[:, idx-1:idx, :] = curr_logit
            pred = torch.cat([pred, curr_pred], dim=1)

        return logit
        

    #here to be some conditional stats for generative forwards
    def forward(self, x, y, is_generative=False):
        if is_generative:
            return self.generative_forward(x, y)
        return self.standard_forward(x, y)



    def cache_generate(self, x, y=None):

        batch_size = x.size(0)
        max_len = self.max_len if y is None else y.size(1)


        pred = torch.zeros((batch_size, 1), dtype=torch.long)
        pred = pred.fill_(self.bos_id).to(self.device)


        cache = None
        e_mask = self.pad_mask(x)
        memory = self.encode(x, e_mask)


        for idx in range(1, max_len):
            y = pred[:, :idx]
            d_out, cache = self.decode(y, memory, cache, e_mask, use_cache=True)
            last_token = self.generator(d_out[:, -1:, :]).argmax(dim=-1)
            pred = torch.cat([pred, last_token], dim=1)
            
            if (pred == self.eos_id).sum().item() == batch_size:
                break

        return pred

