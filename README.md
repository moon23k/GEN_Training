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

자연어생성을 위한 Transformer Seq2seq 모델의 가장 기본적인 학습방식.
학습과정에서 Masking을 통한 Teacher Forcing으로 학습의 안정성을 도모.

<br> 

> **Generative Training**

Teacher Forcing없이 Self Auto Regreesive한 방식의 학습방식.
추론과 동일한 로직으로 학습과정을 전개시킴.
다만 생성 과정에서의 효율성을 증대시키기 위해 cache를 활용합니다.


<br><br> 

## Experimental Setups

| Data Setup | Model Setup | Training Setup |
|---|---|---|
||||

<br><br> 

## Results

| Model Type | Translation | Dialogue | Summarization |
|---|---|---|---|
| Base Line  | - | - | - |
| Scaled Up  | - | - | - |
| Generative | - | - | - |

<br><br> 

## How to use
```
git clone
```

```
python3 setup.py -task ['all', 'translation', 'dialogue', 'summarization']
```

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
