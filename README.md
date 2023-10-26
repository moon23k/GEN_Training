## Generative Training
&nbsp; In Transformer models for natural language generation, Exposure Bias is a significant factor that impairs actual inference performance. 
What makes overcoming Exposure Bias challenging is the use of Teacher Forcing in the typical training process of Transformers.

To address this issue, there are two primary approaches.
The first approach involves increasing the model's size and training it with a substantial amount of data. 
However, this approach may be hindered by data collection challenges and the excessive use of computing resources.
The second approach adopts generative learning. 
This method can be highly effective when training the model with limited data, providing a way to alleviate constraints related to data acquisition.

In this project, we compare the effectiveness of generative learning and data acquisition when training the model with a small dataset to determine which approach is more efficient.

<br><br> 

## Experimental Setups

| Data Setup | Model Setup | Training Setup |
|---|---|---|
||||

<br><br> 

## Results

| Training Method | Translation | Dialogue | Summarization |
|---|---|---|---|
| Basic ||||
| Generative ||||
| Large ||||

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
               -train_opt ['basic', 'generative', 'large']
               -search(Optional) ['greedy', 'beam']
```

<br><br> 

## Reference
* [**Attention Is All You Need**]()

<br> 
