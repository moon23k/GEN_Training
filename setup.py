import os, json
from datasets import load_dataset
from transformers import LEDTokenizerFast



#Select and Tokenize Data
def process_data(orig_data, tokenizer):

    processed = []
    cnt, volumn = 0, 34000
    min_len, max_len = 1000, 3000

    
    for elem in orig_data:
        src, trg = elem['article'].lower(), elem['highlights'].lower()

        #Filter too Short or too Long Context
        if not (min_len < len(src) < max_len):
            continue

        processed.append({"input_ids": tokenizer(src).input_ids,
                          'labels': tokenizer(trg).input_ids})
        
        cnt += 1
        if cnt == volumn:
            break

    return processed



def save_data(data_obj):
    #split data into train/valid/test sets
    train, valid, test = data_obj[:-4000], data_obj[-4000:-1000], data_obj[-1000:]
    data_dict = {k:v for k, v in zip(['train', 'valid', 'test'], [train, valid, test])}

    for key, val in data_dict.items():
        with open(f'data/{key}.json', 'w') as f:
            json.dump(val, f)                    



def main():
    orig = load_dataset('cnn_dailymail', '3.0.0', split='train')
    
    tokenizer = BertTokenizerFast.from_pretrained('allenai/led-base-16384')
    tokenizer.model_max_length = 4098

    processed = process_data(orig, tokenizer)    

    save_data(processed)



if __name__ == '__main__':
    main()