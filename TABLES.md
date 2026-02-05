# MultiNERD results

English only subset of MultiNERD

| Split   | Samples |
| ------- | ------- |
| Train   | 2842119 |
| Eval    | 379011  |
| Test    | 347025  |

**System prompt**:

> You are a named entity recognition model.<br>
> Given a target word and a sentence containing that word, predict the NER tag of the target word based on its context.<br>
> <br>
> Example<br>
> Word: Paris<br>
> Sentence: Paris is the capital of France.<br>
> Output: B-LOC

**Task prompt**:

> \[CLS\] \<system-prompt\> \[SEP\] \<word\> \[SEP\] \<sentence-with-word\> \[SEP\]


## Exact parameters

| Model                              | Parameters | Head parameters | Prefix parameters |
| ---------------------------------- | ---------- | --------------- | ----------------- |
| distilbert/distilbert-base-uncased | 67018783   | 614431          | 41472             |
| jhu-clsp/mmBERT-small              | 140653471  |                 |                   |
| jhu-clsp/mmBERT-base               | 307554079  | 23839           | 44544             |
| openai-community/gpt2              |            |                 |                   |
| openai-community/gpt2-medium       |            |                 |                   |
| openai-community/gpt2-large        |            |                 |                   |
| openai-community/gpt2-xl           |            |                 |                   |
| google-t5/t5-small                 |            |                 |                   |
| google-t5/t5-base                  |            |                 |                   |
| google-t5/t5-large                 |            |                 |                   |
| google-t5/t5-3b                    |            |                 |                   |
| google-t5/t5-11b                   |            |                 |                   |

## Human readable parameters

| Model           | Parameters | Head parameters | Prefix parameters |
| --------------- | ---------- | --------------- | ----------------- |
| distilbert-base | 67M        | 614k            | 41k               |
| mmBERT-small    | 140M       |                 |                   |
| mmBERT-base     | 307M       | 23k             | 44k               |
| gpt2            | 124M       |                 |                   |
| gpt2-medium     | 355M       |                 |                   |
| gpt2-large      | 774M       |                 |                   |
| gpt2-xl         | 1.5B       |                 |                   |
| t5-small        | 60M        |                 |                   |
| t5-base         | 220M       |                 |                   |
| t5-large        | 770M       |                 |                   |
| t5-3b           | 3B         |                 |                   |
| t5-11b          | 11B        |                 |                   |

## Test metrics

### Full fine-tune

| Model           | Accuracy | Precision | Recall | F1     |
| --------------- | -------- | --------- | ------ | ------ |
| distilbert-base | 0.9791   | 0.7130    | 0.6621 | 0.6852 |
| mmBERT-small    |          |           |        |        |
| mmBERT-base     | 0.9872   | 0.8276    | 0.7552 | 0.7853 |
| gpt2            |          |           |        |        |
| gpt2-medium     |          |           |        |        |
| gpt2-large      |          |           |        |        |
| gpt2-xl         |          |           |        |        |
| t5-small        |          |           |        |        |
| t5-base         |          |           |        |        |
| t5-large        |          |           |        |        |
| t5-3b           |          |           |        |        |
| t5-11b          |          |           |        |        |

### Full fine-tune with system propmt

| Model           | Accuracy | Precision | Recall | F1     |
| --------------- | -------- | --------- | ------ | ------ |
| distilbert-base | 0.9764   | 0.6899    | 0.6114 | 0.6399 |
| mmBERT-small    |          |           |        |        |
| mmBERT-base     | 0.9875   | 0.8160    | 0.7499 | 0.7754 |
| gpt2            |          |           |        |        |
| gpt2-medium     |          |           |        |        |
| gpt2-large      |          |           |        |        |
| gpt2-xl         |          |           |        |        |
| t5-small        |          |           |        |        |
| t5-base         |          |           |        |        |
| t5-large        |          |           |        |        |
| t5-3b           |          |           |        |        |
| t5-11b          |          |           |        |        |

### Full fine-tune with random system propmt

| Model           | Accuracy | Precision | Recall | F1     |
| --------------- | -------- | --------- | ------ | ------ |
| distilbert-base | 0.9795   | 0.7161    | 0.6670 | 0.6861 |
| mmBERT-small    |          |           |        |        |
| mmBERT-base     | 0.9874   | 0.8506    | 0.7678 | 0.8004 |
| gpt2            |          |           |        |        |
| gpt2-medium     |          |           |        |        |
| gpt2-large      |          |           |        |        |
| gpt2-xl         |          |           |        |        |
| t5-small        |          |           |        |        |
| t5-base         |          |           |        |        |
| t5-large        |          |           |        |        |
| t5-3b           |          |           |        |        |
| t5-11b          |          |           |        |        |

### Head fine-tune

| Model           | Accuracy | Precision | Recall | F1     |
| --------------- | -------- | --------- | ------ | ------ |
| distilbert-base | 0.9198   | 0.4439    | 0.1532 | 0.1855 |
| mmBERT-small    |          |           |        |        |
| mmBERT-base     | 0.9092   | 0.4256    | 0.1457 | 0.1848 |
| gpt2            |          |           |        |        |
| gpt2-medium     |          |           |        |        |
| gpt2-large      |          |           |        |        |
| gpt2-xl         |          |           |        |        |
| t5-small        |          |           |        |        |
| t5-base         |          |           |        |        |
| t5-large        |          |           |        |        |
| t5-3b           |          |           |        |        |
| t5-11b          |          |           |        |        |

### Head fine-tune with system propmt

| Model           | Accuracy | Precision | Recall | F1     |
| --------------- | -------- | --------- | ------ | ------ |
| distilbert-base | 0.8801   | 0.1379    | 0.0554 | 0.0647 |
| mmBERT-small    |          |           |        |        |
| mmBERT-base     | 0.9118   | 0.3967    | 0.1342 | 0.1645 |
| gpt2            |          |           |        |        |
| gpt2-medium     |          |           |        |        |
| gpt2-large      |          |           |        |        |
| gpt2-xl         |          |           |        |        |
| t5-small        |          |           |        |        |
| t5-base         |          |           |        |        |
| t5-large        |          |           |        |        |
| t5-3b           |          |           |        |        |
| t5-11b          |          |           |        |        |

### Head fine-tune with random system propmt

| Model           | Accuracy | Precision | Recall | F1     |
| --------------- | -------- | --------- | ------ | ------ |
| distilbert-base | 0.8759   | 0.1472    | 0.0443 | 0.0500 |
| mmBERT-small    |          |           |        |        |
| mmBERT-base     | 0.8963   | 0.2602    | 0.0950 | 0.1189 |
| gpt2            |          |           |        |        |
| gpt2-medium     |          |           |        |        |
| gpt2-large      |          |           |        |        |
| gpt2-xl         |          |           |        |        |
| t5-small        |          |           |        |        |
| t5-base         |          |           |        |        |
| t5-large        |          |           |        |        |
| t5-3b           |          |           |        |        |
| t5-11b          |          |           |        |        |

### Prompt-tune random initialization

| Model           | Accuracy | Precision | Recall | F1     |
| --------------- | -------- | --------- | ------ | ------ |
| distilbert-base | 0.9653   | 0.7005    | 0.4295 | 0.4897 |
| mmBERT-small    |          |           |        |        |
| mmBERT-base     | 0.9603   | 0.5415    | 0.3454 | 0.3933 |
| gpt2            |          |           |        |        |
| gpt2-medium     |          |           |        |        |
| gpt2-large      |          |           |        |        |
| gpt2-xl         |          |           |        |        |
| t5-small        |          |           |        |        |
| t5-base         |          |           |        |        |
| t5-large        |          |           |        |        |
| t5-3b           |          |           |        |        |
| t5-11b          |          |           |        |        |

### Prompt-tune with pretrained model embedding initialization

| Model           | Accuracy | Precision | Recall | F1     |
| --------------- | -------- | --------- | ------ | ------ |
| distilbert-base | 0.9739   | 0.7124    | 0.5686 | 0.6094 |
| mmBERT-small    |          |           |        |        |
| mmBERT-base     | 0.9828   | 0.8180    | 0.6658 | 0.7132 |
| gpt2            |          |           |        |        |
| gpt2-medium     |          |           |        |        |
| gpt2-large      |          |           |        |        |
| gpt2-xl         |          |           |        |        |
| t5-small        |          |           |        |        |
| t5-base         |          |           |        |        |
| t5-large        |          |           |        |        |
| t5-3b           |          |           |        |        |
| t5-11b          |          |           |        |        |
