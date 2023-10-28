## Generative Training
&nbsp; The Transformer Seq2Seq model for natural language generation typically employs Teacher Forcing during training. 
While this approach promotes stable learning, it has limitations in improving inference performance since Teacher Forcing is absent during the inference phase.

To address this issue, there are two main approaches. 
One involves increasing the amount of data to enhance the model's generalization performance. 
The other approach leverages the same mechanisms used during inference in the training phase to improve inference capabilities.

In this project, we aim to directly assess which of the two methods, increasing the data volume or adopting generative learning, is more effective for three natural language generation tasks. 
This evaluation is conducted in a scenario where the available data is limited.

<br><br> 

## Training Methodologies

> **Teacher Forcing Training**

Teacher forcing training is the most fundamental training method for Transformer Seq2Seq models used in natural language generation. 
During the training process, it ensures the stability of learning through Teacher Forcing, employing masking.

<br> 

> **Generative Training**

Generative Training is a training approach that follows a self-auto-regressive method without Teacher Forcing. 
However, to enhance efficiency during the generation process, it utilizes caching.

<br><br> 

## Experimental Setups

| Data Setup | Model Setup | Training Setup |
|---|---|---|
||||

<br><br> 

## Results

| Model Type | Translation | Dialogue | Summarization |
|---|---|---|---|
| Base Line  |  3.29 | - | - |
| Augmented  | 12.87 | - | - |
| Generative |  0.0  | - | - |

<br><br> 

## How to use

**Clone git on your local env**
```
git clone https://github.com/moon23k/GEN_Training.git
```

**Setup Dataset and Tokenizer via setup.py file**
```
python3 setup.py -task ['all', 'translation', 'dialogue', 'summarization']
```

**Actual Process via run.py file**
```
python3 run.py -task ['translation', 'dialogue', 'summarization']
               -mode ['train', 'test', 'inference']
               -model ['baseline', 'scale_up', 'generative']
               -search(Optional) ['greedy', 'beam']
```

<br><br> 

## Reference
* [**Attention Is All You Need**]()

<br> 
