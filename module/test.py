import torch, evaluate



class Tester:
    def __init__(self, config, model, tokenizer, test_dataloader):
        super(Tester, self).__init__()
        
        self.model = model
        self.tokenizer = tokenizer
        self.dataloader = test_dataloader

        self.bos_id = config.bos_id
        self.eos_id = config.eos_id
        self.device = config.device
        self.max_len = config.max_len
        
        self.metric_module = evaluate.load('BLEU')
        


    def tokenize(self, batch):
        return [self.tokenizer.decode(x) for x in batch.tolist()]


    def predict(self, x):

        batch_size = x.size(0)
        pred = torch.zeros((batch_size, 1), dtype=torch.long)
        pred = pred.fill_(self.bos_id).to(self.device)

        e_mask = self.model.pad_mask(x)
        memory = self.model.encode(x, e_mask)

        for idx in range(1, self.max_len+1):
            y = pred[:, :idx]
            d_out, cache = self.model.decode(
                y, memory, cache=None, e_mask=e_mask, 
                d_mask=None, use_cache=False
            )

            curr_logit = self.model.generator(d_out[:, -1:, :])
            curr_pred = curr_logit.argmax(dim=-1)

            pred = torch.cat([pred, curr_pred], dim=1)

            #Early Stop Condition
            if (pred == self.eos_id).sum().item() == batch_size:
                break

        return pred



    def evaluate(self, pred, label):
        score = self.metric_module.compute(
            predictions=pred, 
            references =[[l] for l in label]
        )['bleu']


        return score * 100



    def test(self):
        score = 0.0         
        self.model.eval()

        with torch.no_grad():
            for batch in self.dataloader:
                x = batch['x'].to(self.device)
                y = self.tokenize(batch['y'])

                pred = self.predict(x)
                pred = self.tokenize(pred)
                
                score += self.evaluate(pred, y)

        txt = f"TEST Result on {self.model_type.upper()} model"
        txt += f"\n-- Score: {round(score/len(self.dataloader), 2)}\n"
        print(txt)        