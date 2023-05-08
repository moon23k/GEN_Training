import time, json, torch
import torch.nn as nn
import torch.nn.functional as F
import torch.amp as amp
import torch.optim as optim



class TrainerBase:
    def __init__(self, config):
        
        self.mode = config.mode
        self.clip = config.clip
        self.device = config.device
        self.max_len = config.max_len
        self.n_epochs = config.n_epochs
        self.device_type = config.device_type
        self.scaler = torch.cuda.amp.GradScaler()
        self.iters_to_accumulate = config.iters_to_accumulate

        self.early_stop = config.early_stop
        self.patience = config.patience


    @staticmethod
    def measure_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_min = int(elapsed_time / 60)
        elapsed_sec = int(elapsed_time - (elapsed_min * 60))
        return f"{elapsed_min}m {elapsed_sec}s"


    def collate_dis_inputs(self, uttr, resp, pred):

        pos_encodings = self.tokenize(uttr + resp)
        neg_encodings = self.tokenize(uttr + pred)

        pos_ids, pos_mask = pos_encodings.input_ids, pos_encodings.attention_mask
        neg_ids, neg_mask = neg_encodings.input_ids, neg_encodings.attention_mask

        batch_size, pos_len = pos_ids.shape
        neg_len = neg_ids.size(1)
        pad_len = neg_len - pos_len

        #Padding
        if pad_len > 0:
            pad_tensor = torch.zeros((batch_size, pad_len), dtype=torch.long).to(self.device)
            pos_ids = torch.cat((pos_ids, pad_tensor), dim=1)
            pos_mask = torch.cat((pos_mask, pad_tensor), dim=1)

        else:
            pad_tensor = torch.zeros((batch_size, -pad_len), dtype=torch.long).to(self.device)
            neg_ids = torch.cat((neg_ids, pad_tensor), dim=1)
            neg_mask = torch.cat((neg_mask, pad_tensor), dim=1)


        ids = torch.cat((pos_ids, neg_ids), dim=0).to(self.device)
        masks = torch.cat((pos_mask, neg_mask), dim=0).to(self.device)

        labels = torch.cat((torch.zeros(batch_size), torch.ones(batch_size)), dim=0)
        indice = torch.randperm(batch_size * 2).long()

        #Shuffle Discriminator inputs
        ids = ids[indice].to(self.device)
        masks = masks[indice].to(self.device)
        labels = labels[indice].to(self.device)

        return ids, masks, labels


    def save_ckpt(self, epoch, ckpt, model, optimizer):
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                    ckpt)

        


class Trainer(TrainerBase):
    def __init__(self, config, g_model, d_model, train_dataloader, valid_dataloader):
        super(Trainer, self).__init__(config)

        self.g_model = g_model
        self.d_model = d_model

        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        
        self.g_optimizer = optim.AdamW(params=self.g_model.parameters(), lr=config.lr)
        self.d_optimizer = optim.AdamW(params=self.d_model.parameters(), lr=config.lr)

        self.g_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.g_optimizer, 'min')
        self.d_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.d_optimizer, 'min')

        self.g_ckpt = config.g_ckpt
        self.d_ckpt = config.d_ckpt

        self.record_path = 'ckpt/train.json'
        self.record_keys = ['epoch', 'g_train_loss', 'd_train_loss', 'g_valid_loss', 
                            'd_valid_loss', 'g_lr', 'd_lr', 'epoch_time']



    def print_epoch(self, record_dict):
        print(f"""Epoch {record_dict['epoch']}/{self.n_epochs} | \
              Time: {record_dict['epoch_time']}""".replace(' ' * 14, ''))

        print(f"""  >> Generator Train Loss: {record_dict['g_train_loss']:.3f}     | \
              Generator Valid Loss: {record_dict['g_valid_loss']:.3f}""".replace(' ' * 14, ''))

        print(f"""  >> Discriminator Train Loss: {record_dict['d_train_loss']:.3f} | \
              Discriminator Valid Loss: {record_dict['d_valid_loss']:.3f}\n""".replace(' ' * 14, ''))



    def discriminate(self, uttr, resp=None, pred=None):
        if resp is None:
            encodings = self.tokenize(uttr + pred)
            ids = encodings.input_ids
            masks = encodings.attention_mask
            labels = torch.zeros(ids.size(0)).to(self.device)
        else:
            ids, masks, labels = self.collate_dis_inputs(uttr, resp, pred)

        with torch.autocast(device_type=self.device_type, dtype=torch.float16):
            outs = self.d_model(input_ids=ids, attention_mask=masks, labels=labels)        

        return outs


    def generate(self, text, text_mask):
        is_training = self.g_model.training()
        
        if is_training:
            self.g_model.eval()

        with torch.autocast(device_type=self.device.type, dtype=torch.float16):
            pred = self.g_model.generate(input_ids=text,
                                         attention_mask=text_mask, 
                                         max_new_tokens=self.max_len, 
                                         use_cache=True)    
        
        if is_training:
            self.g_model.train()
            
        return pred


    def get_losses(self, batch):

        #Collate Batch
        text = batch['text'].to(self.device)
        summ = batch['summ'].to(self.device)
        batch_size = text.size(0)


        #Generate Pred
        pred = self.g_model.generate(input_ids=text, 
                                     attention_mask=(text == self.pad_id).to(self.device))


        #manipulate discriminator_inputs
        dis_inputs = []
        for i in range(batch_size):
            dis_inputs.append(torch.cat((text[i] != self.pad_id), (summ[i] != self.pad_id)))
            dis_inputs.append(torch.cat((text[i] != self.pad_id), (pred[i] != self.pad_id)))
        dis_inputs = pad_sequence(dis_inputs, batch_first=True, padding_value=self.pad_id)

        #get g_loss
        g_logit = self.d_model(dis_inputs[::2]).logit
        g_loss = (g_logit.softmax >= 0.5).sum().item() / batch_size
        
        if not g_logit:
            g_logit = 1e-4

        g_loss = -torch.log(torch.tensor(g_logit, requires_grad=True)).to(self.device)

        #Shuffle
        indice = torch.randperm(batch_size * 2).long()
        labels = torch.tensor([0 if i % 2 == 0 else 1 for i in range(batch_size*2)])
        dis_inputs = dis_inputs[indice]
        labels = labels[indice]

        d_loss = self.d_model(dis_inputs, labels).loss
        
        return g_loss, d_loss


    def train(self):
        records = []
        patience = self.patience
        prev_loss, g_best_loss, d_best_loss = float('inf'), float('inf'), float('inf')

        for epoch in range(1, self.n_epochs + 1):
            start_time = time.time()

            record_vals = [epoch, *self.train_epoch(), *self.valid_epoch(), 
                           self.g_optimizer.param_groups[0]['lr'],
                           self.d_optimizer.param_groups[0]['lr'],
                           self.measure_time(start_time, time.time())]

            record_dict = {k: v for k, v in zip(self.record_keys, record_vals)}
            
            records.append(record_dict)
            self.print_epoch(record_dict)
            
            g_curr_loss = record_dict['g_valid_loss']
            d_curr_loss = record_dict['d_valid_loss']

            self.g_scheduler.step(g_curr_loss)
            self.d_scheduler.step(d_curr_loss)


            #save best discriminator states
            if d_best_loss >= d_curr_loss:
                d_best_loss = d_curr_loss
                self.save_ckpt(epoch, self.d_ckpt, self.d_model, self.d_optimizer)


            #save best generator states
            if g_best_loss >= g_curr_loss:
                g_best_loss = g_curr_loss
                self.save_ckpt(epoch, self.g_ckpt, self.g_model, self.g_optimizer)

            #Early Stopping Process
            if self.early_stop:
                if prev_loss > g_curr_loss:
                    patience = self.patience
            
                else:
                    patience -= 1
                    if not patience:
                        print('--- Training Ealry Stopped ---\n')
                        break

                prev_loss = g_curr_loss


        #save train_records
        with open(self.record_path, 'w') as fp:
            json.dump(records, fp)        
            


    def train_epoch(self):
        g_epoch_loss, d_epoch_loss = 0, 0
        tot_len = len(self.train_dataloader)
        
        self.g_model.train()
        self.d_model.train()


        for idx, batch in enumerate(self.train_dataloader):
            
            idx += 1
            g_loss, d_loss = self.get_losses(batch)
            g_loss = g_loss / self.iters_to_accumulate
            d_loss = d_loss / self.iters_to_accumulate
            

            g_loss.backward()
            self.scaler.scale(d_loss).backward()

            if (idx % self.iters_to_accumulate == 0) or (idx == tot_len):
                self.scaler.unscale_(self.g_optimizer)
                self.scaler.unscale_(self.d_optimizer)

                #Gradient Clipping
                nn.utils.clip_grad_norm_(self.g_model.parameters(), max_norm=self.clip)
                nn.utils.clip_grad_norm_(self.d_model.parameters(), max_norm=self.clip)
                
                #Gradient Update & Scaler Update
                self.g_optimizer.step()
                self.d_optimizer.step()
                
                self.scaler.step(self.g_optimizer)
                self.scaler.step(self.d_optimizer)
                
                self.scaler.update()
                
                self.g_optimizer.zero_grad()
                self.d_optimizer.zero_grad()

            g_epoch_loss += g_loss.item()
            d_epoch_loss += d_loss.item()
        
    
        return round(g_epoch_loss / tot_len, 3), round(d_epoch_loss / tot_len, 3)
    


    def valid_epoch(self):
        g_epoch_loss, d_epoch_loss = 0, 0
        tot_len = len(self.valid_dataloader)

        self.g_model.eval()
        self.d_model.eval()
        
        with torch.no_grad():
            for batch in self.valid_dataloader:          
                g_loss, d_loss = self.get_losses(batch)

                g_epoch_loss += g_loss.item()
                d_epoch_loss += d_loss.item()


        return round(g_epoch_loss / tot_len, 3), round(d_epoch_loss / tot_len, 3)
