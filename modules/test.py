import time, torch



class Tester:
    def __init__(self, config, g_model, d_model, 
                 g_tokenizer, d_tokenizer, test_dataloader):

        self.g_model = g_model
        self.d_model = d_model
        
        self.g_tokenizer = g_tokenizer
        self.d_tokenizer = d_tokenizer
        
        self.device = config.device
        self.max_len = config.max_len
        self.dataloader = test_dataloader


    @staticmethod
    def measure_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_min = int(elapsed_time / 60)
        elapsed_sec = int(elapsed_time - (elapsed_min * 60))
        return f"{elapsed_min}m {elapsed_sec}s"        


    def tokenize(self, tokenizer, tokenizer_inputs):
        return tokenizer(tokenizer_inputs, 
                         padding=True, 
                         truncation=True, 
                         return_tensors='pt').to(self.device)


    def test(self):
        scores = 0

        self.g_model.eval()
        self.d_model.eval()

        with torch.no_grad():
            for idx, batch in enumerate(self.dataloader):   
                uttr, resp = batch[0], batch[1]

                #tokenize inputs for generator
                g_uttr_encodings = self.tokenizer(self.g_tokenizer, uttr)
                g_ids = g_uttr_encodings.input_ids
                g_masks = g_uttr_encodings.attention_mask

                #generate predictions
                preds = self.g_model.generate(input_ids=g_ids,
                                              attention_mask=g_masks, 
                                              max_new_tokens=self.max_len, 
                                              use_cache=True)
                #Decode generator predictions
                preds = self.g_tokenizer.batch_decode(preds, skip_special_tokens=True)

                #Tokenize inputs for discriminator
                d_encodings = self.tokenize(self.d_tokenizer, preds)
                d_ids = d_encodings.input_ids
                d_masks = d_encodings.attention_mask
                logits = self.d_model(input_ids=d_ids, attention_mask=d_masks)
                scores += logits[logits > 0.5].sum()

        scores = scores / len(dataloader)

        print('Test Results')
        print(f"  >> Test Score: {scores:.2f}")
