# BERT-NER
This is a named entity recognizer based on [pytorch-pretrained-bert](https://github.com/huggingface/pytorch-pretrained-BERT).
## Requirements
- python 3.5+
- pytorch 0.4.1
- pytorch-pretrained-bert 0.4.0
- tqdm
- PyYAML
## Overview
- njuner
  - An Chinese NER package, more details in section [NJUNER](#NJUNER).
- preprocess_msra.py
  - A tool for preprocessing the MSRA NER dataset.
- preprocess_pd98.py
  - A tool for preprocessing the [People's Daily 1998.1 dataset](https://github.com/hankcs/OpenCorpus/tree/master/pku98).
- run_ner.py
  - A tool for training and testing bert-ner model on different datasets.
  - Working with task_config.yaml.
## NJUNER
An Chinese NERer which recognizes PERSONs, LOCATIONs and ORGANIZATIONs in texts. It is completely character-based and does not require word segmentation or part-of-speech information.
### Installation
```bash
pip install njuner
```
### Usage
- As a package
  ```python
  from njuner import NJUNER
  ner = NJUNER(model_dir=model_path)
  ner.label(['李雷和韩梅梅去上海迪斯尼乐园。'])
  # [[('B-PER', '李'), ('I-PER', '雷'), ('O', '和'), ('B-PER', '韩'), ('I-PER', '梅'), ('I-PER', '梅'), ('O', '去'), ('B-ORG', '上'), ('I-ORG', '海'), ('I-ORG', '迪'), ('I-ORG', '斯'), ('I-ORG', '尼'), ('I-ORG', '乐'), ('I-ORG', '园'), ('O', '。') ]]
  ```
- As a command line tool
  - Manual
    ```bash
    njuner -h
    ```
  - An example
    ```bash
    njuner --model_dir model_path --input_file input.txt --output_dir ./
    ``` 
    This will produce there files, which are "tokens.txt", "predictions.txt" and "summary.txt", in the output directory.
- Pretrained model
  
  You can get the model pretrained on the MSRA NER dataset from the [NJUNER releases page](https://github.com/ericput/bert-ner/releases). Uncompress the model archive and pass the directory to the parameter "model_dir".
### Performance
### Metrics: Span-based F1
- Training and testing on the corresponding dataset.
  
  |Item|MSRA|Weibo-NE|
  |-|-|-|
  |SOTA|93.18|55.28|
  |NJUNER|94.78|66.95|
  - Results of SOTA are according to the paper [Chinese NER Using Lattice LSTM](http://aclweb.org/anthology/P18-1144).
  - Our model fined tune on the BERT, which pretrained on large-scale unlabeled corpus, so the above results are not strictly comparable.

- Comparison of different Chinese NER tools.

  |Item|MSRA|Weibo-NE|
  |-|-|-|
  |[HanLP](https://github.com/hankcs/HanLP)|72.65|38.66|
  |[LTP](https://github.com/HIT-SCIR/pyltp)|73.34|43.97|
  |NJUNER|81.58|63.08|

  - The NER modules of HanLP and LTP are both trained on the People's Daily 1998.1 dataset. For comparison, NJUNER is also trained on the same dataset and achieve the above results on MSRA and Weibo-NE respectively. Also, their target entity types are same with our tool's, which are "PER", "LOC" and "ORG".
  - There is another entity type "GPE" in Weibo-NE dataset. For comparison, we uniformly refer to "GPE" as "LOC".
