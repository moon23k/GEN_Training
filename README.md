## Intelli GEN
&nbsp; This repository contains a series of code aimed at enhancing performance through **Intelligent Generative Training** using Transformer model. Typically, Transformers are well-suited for large-scale parallel processing training based on attention mechanism, employing Teacher Forcing during this process. However, during actual inference, they rely solely on predictions from the model's previous time step. This creates a discrepancy between the training and inference processes, making it challenging to improve inference performance. However, applying the same logic during training as in inference would negate the advantages of Transformers. Therefore, there is a need for a more intelligent approach to generative learning. Detailed training strategies are discussed below.

<br><br> 

## Fine-Tuning Strategy

### 1.&nbsp; Standard&hairsp; Fine-Tuning

> Teacher forcing training is the most fundamental training method for Transformer Seq2Seq models used in natural language generation. 
During the training process, it ensures the stability of learning through Teacher Forcing, employing masking.

<br> 

### 2.&nbsp; Generative&hairsp; Fine-Tuning

> Generative Training is a training approach that follows a self-auto-regressive method without Teacher Forcing. 
However, to enhance efficiency during the generation process, it utilizes caching.

<br> 

### 3.&nbsp; Slow Sequence GAN&hairsp; Fine-Tuning

> Generative Training is a training approach that follows a self-auto-regressive method without Teacher Forcing. 
However, to enhance efficiency during the generation process, it utilizes caching.

<br> 

### 4.&nbsp; Intelli GEN&hairsp; Fine-Tuning

> Generative Training is a training approach that follows a self-auto-regressive method without Teacher Forcing. 
However, to enhance efficiency during the generation process, it utilizes caching.

<br><br> 

## Experimental Setups

| Data Setup | Model Setup | Training Setup |
| --- | --- | --- |
|`Dataset:` WMT14 En-De |`Architecture:` Transformer |`Num Epochs:` 10 |

<br><br> 

## Results

| Strategy | BLUE Score | Epoch Time |
|---|---|---|
| Baseline     | - | - |
| Standard Fine-Tuning   | - | - |
| Generative Fine-Tuning | - | - |
| SlowGAN Fine-Tuning | - | - |
| IntelliGEN Fine-Tuning | - | - |

<br><br> 

## How to use

**Clone git on your local env**
```
git clone https://github.com/moon23k/GEN_Training.git
```

**Setup Dataset and Tokenizer via setup.py file**
```
python3 setup.py
```

**Actual Process via run.py file**
```
python3 run.py -mode ['train', 'finetune', 'test', 'inference']
               -strategy ['std', 'gen', 'slow', 'intelli']
               -search(Optional) ['greedy', 'beam']
```

<br><br> 

## Reference
* [**Attention Is All You Need**]()

<br> 
