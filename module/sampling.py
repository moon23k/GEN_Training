import json, torch
from module import load_dataloader



class Sampler(object):
	def __init__(self, config, generator, tokenizer):

		self.generator = generator

		self.train_dataloader = load_dataloader(config, tokenizer, 'train')
		self.valid_dataloader = load_dataloader(config, tokenizer, 'valid')
		self.test_dataloader = load_dataloader(config, tokenizer, 'test')

		self.sample_train_ckpt = 'data/sample_train.json'
		self.sample_valid_ckpt = 'data/sample_valid.json'
		self.sample_tes_ckpt = 'data/sample_valid.json'



	def generate_sample(self, dataloader):
		samples = []

		with torch.no_grad():
			for batch in enumerate(dataloader):
				x, label = batch['x'].to(self.device), batch['y'].to(self.device)
				
				with torch.autocast(device_type=self.device_type, dtype=torch.float16):
					pred = self.generator.generate(x)

				for p, l in zip(pred, label):
					samples.append({'x': p, 'y': 0})
					samples.append({'x': l, 'y': 1})				

		return samples


	def save_sample(self, sample, sample_ckpt):
		with open(sample_ckpt, 'w') as f:
			json.dump(sample_ckpt, f)


	def sample(self):
		train_samples = self.generate_sample(self.train_dataloader)
		valid_samples = self.generate_sample(self.valid_dataloader)
		test_samples = self.generate_sample(self.test_dataloader)

		self.save_sample(train_samples, self.sample_train_ckpt)
		self.save_sample(valid_samples, self.sample_valid_ckpt)
		self.save_sample(test_samples, self.sample_test_ckpt)
