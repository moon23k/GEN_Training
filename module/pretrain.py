import time, json, torch
import torch.nn as nn
import torch.amp as amp
import torch.optim as optim
from module.train import TrainerBase




class GenTrainer(TrainerBase):
    def __init__(self, config, model, train_dataloader, valid_dataloader):
        
        super(GenTrainer, self).__init__(config)

        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')

        self.ckpt = config.g_base_ckpt
        self.record_path = "ckpt/generator_base.json"
        self.record_keys = ['epoch', 'train_loss', 'valid_loss', 'lr', 'train_time']


    def print_epoch(self, record_dict):
        print(f"""Epoch {record_dict['epoch']}/{self.n_epochs} | \
              Time: {record_dict['train_time']}""".replace(' ' * 14, ''))

        print(f"""  >> Generator Train Loss: {record_dict['train_loss']:.3f} | \
              Generator Valid Loss: {record_dict['valid_loss']:.3f}\n""".replace(' ' * 14, ''))        



    def train(self):
        records = []
        patience = self.patience
        prev_loss, best_loss = float('inf'), float('inf')


        for epoch in range(1, self.n_epochs + 1):
            start_time = time.time()
            record_vals = [epoch, self.train_epoch(), self.valid_epoch(), 
                           self.optimizer.param_groups[0]['lr'],
                           self.measure_time(start_time, time.time())]
            record_dict = {k: v for k, v in zip(self.record_keys, record_vals)}
            
            records.append(record_dict)
            self.print_epoch(record_dict)
            
            val_loss = record_dict['valid_loss']
            self.scheduler.step(val_loss)

            #save best model
            if best_loss > val_loss:
                best_loss = val_loss
                torch.save({'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict()},
                            self.ckpt)
            
            #Early Stopping Process
            if self.early_stop:
                if prev_loss > val_loss:
                    patience = self.patience
            
                else:
                    patience -= 1
                    if not patience:
                        print('--- Training Ealry Stopped ---\n')
                        break

                prev_loss = val_loss


        #save train_records
        with open(self.record_path, 'w') as fp:
            json.dump(records, fp)    



    def train_epoch(self):
        epoch_loss = 0
        tot_len = len(self.train_dataloader)
        self.model.train()
        
        for idx, batch in enumerate(self.train_dataloader):
            idx += 1
            uttr, resp = batch[0], batch[1]

            uttr_encodings = self.tokenize(uttr)
            resp_encodings = self.tokenize(resp)

            with torch.autocast(device_type=self.device_type, dtype=torch.float16):
                loss = self.model(input_ids=uttr_encodings.input_ids,
                                  attention_mask=uttr_encodings.attention_mask,
                                  labels=resp_encodings.input_ids).loss
                loss = loss / self.iters_to_accumulate

            #Backward Loss
            self.scaler.scale(loss).backward()        

            if (idx % self.iters_to_accumulate == 0) or (idx == tot_len):
                #Gradient Clipping
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip)
                
                #Gradient Update & Scaler Update
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()                

            epoch_loss += loss.item()

        epoch_loss = round(epoch_loss / tot_len, 3)
        return epoch_loss


    def valid_epoch(self):
        epoch_loss = 0
        self.model.eval()

        for batch in self.valid_dataloader:
            uttr, resp = batch[0], batch[1]
            uttr_encodings = self.tokenize(uttr)
            resp_encodings = self.tokenize(resp)

            with torch.no_grad():
                loss = self.model(input_ids=uttr_encodings.input_ids, 
                                  attention_mask=uttr_encodings.attention_mask,
                                  labels=resp_encodings.input_ids).loss

            epoch_loss += loss.item()

        epoch_loss = round(epoch_loss / len(self.valid_dataloader), 3)
        return epoch_loss




class DisTrainer(TrainerBase):
    def __init__(self, config, model, tokenizer, train_dataloader, valid_dataloader):
        
        super(DisTrainer, self).__init__(config)

        self.model = model
        self.tokenizer = tokenizer

        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

        self.optimizer = optim.AdamW(params=self.model.parameters(), lr=config.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')

        self.ckpt = config.d_base_ckpt
        self.record_path = 'ckpt/discriminator_base.json'
        self.record_keys = ['epoch', 'train_loss', 'valid_loss', 'lr', 'train_time']



    def print_epoch(self, record_dict):
        print(f"""Epoch {record_dict['epoch']}/{self.n_epochs} | \
              Time: {record_dict['train_time']}""".replace(' ' * 14, ''))

        print(f"""  >> Discriminator Train Loss: {record_dict['train_loss']:.3f} | \
              Discriminator Valid Loss: {record_dict['valid_loss']:.3f}\n""".replace(' ' * 14, ''))        



    def train(self):
        records = []        
        prev_loss, best_loss = float('inf'), float('inf')
        patience = self.patience

        for epoch in range(1, self.n_epochs + 1):
            start_time = time.time()
            record_vals = [epoch, self.train_epoch(), self.valid_epoch(), 
                           self.optimizer.param_groups[0]['lr'],
                           self.measure_time(start_time, time.time())]
            record_dict = {k: v for k, v in zip(self.record_keys, record_vals)}

            records.append(record_dict)
            self.print_epoch(record_dict)
            
            val_loss = record_dict['valid_loss']
            self.scheduler.step(val_loss)

            #save best model
            if best_loss > val_loss:
                best_loss = val_loss
                torch.save({'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict()},
                            self.ckpt)
                
            #Early Stopping Process
            if self.early_stop:
                if prev_loss > val_loss:
                    patience = self.patience
            
                else:
                    patience -= 1
                    if not patience:
                        print('--- Training Ealry Stopped ---\n')
                        break

                prev_loss = val_loss

        #save train_records
        with open(self.record_path, 'w') as fp:
            json.dump(records, fp)    



    def train_epoch(self):
        epoch_loss = 0
        tot_len = len(self.train_dataloader)

        self.model.train()
        for idx, batch in enumerate(self.train_dataloader):
            
            idx += 1
            uttr, pos, neg = batch[0], batch[1], batch[2]
            ids, masks, labels = self.collate_dis_inputs(uttr, pos, neg)

            with torch.autocast(device_type=self.device_type, dtype=torch.float16):
                loss = self.model(input_ids=ids, attention_mask=masks, labels=labels).loss            
            loss = loss / self.iters_to_accumulate

            #Backward Loss
            self.scaler.scale(loss).backward()        

            if (idx % self.iters_to_accumulate == 0) or (idx == tot_len):
                #Gradient Clipping
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip)
                
                #Gradient Update & Scaler Update
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()                

            epoch_loss += loss.item()

        epoch_loss = round(epoch_loss / tot_len, 3)
        return epoch_loss
    


    def valid_epoch(self):
        epoch_loss = 0

        self.model.eval()
        for batch in self.valid_dataloader:
            ids, masks, labels = self.collate_batch(batch)
            with torch.no_grad():
                loss = self.model(input_ids=ids, attention_mask=masks, labels=labels).loss
            epoch_loss += loss.item()

        epoch_loss = round(epoch_loss / len(self.valid_dataloader), 3)
        return epoch_loss