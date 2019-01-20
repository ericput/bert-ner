import os
import sys
import re
import codecs
import logging
import random
import yaml
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import PreTrainedBertModel, BertModel
from pytorch_pretrained_bert.optimization import BertAdam

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class BertForNER(PreTrainedBertModel):

    def __init__(self, config, num_labels):
        super(BertForNER, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(0.2)
        self.hidden2label = torch.nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, segment_ids, input_mask, predict_mask=None, one_hot_labels=None):
        bert_layer, _ = self.bert(input_ids, segment_ids, input_mask, output_all_encoded_layers=False)
        if one_hot_labels is not None:
            bert_layer = self.dropout(bert_layer)
        logits = self.hidden2label(bert_layer)

        if one_hot_labels is not None:
            p = torch.nn.functional.softmax(logits, -1)
            losses = -torch.log(torch.sum(one_hot_labels * p, -1))
            losses = torch.masked_select(losses, predict_mask)
            return torch.sum(losses)
        else:
            return logits


class InputExample(object):

    def __init__(self, guid, words, labels):
        self.guid = guid
        self.words = words
        self.labels = labels


class InputFeatures(object):

    def __init__(self, input_ids, input_mask, segment_ids, predict_mask, one_hot_labels):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.predict_mask = predict_mask
        self.one_hot_labels = one_hot_labels


class DataProcessor(object):

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    @staticmethod
    def create_examples_from_conll_format_file(data_file, set_type):
        examples = []
        words = []
        labels = []
        for index, line in enumerate(codecs.open(data_file, encoding='utf-8')):
            if not line.strip():
                guid = "%s-%d" % (set_type, index)
                examples.append(InputExample(guid=guid, words=words, labels=labels))
                words = []
                labels = []
            else:
                segs = line.split()
                words.append(segs[0])
                labels.append(segs[-1])
        return examples

    @staticmethod
    def get_labels():
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class CONLLProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return DataProcessor.create_examples_from_conll_format_file(os.path.join(data_dir, 'train.txt'), 'train')

    def get_dev_examples(self, data_dir):
        return DataProcessor.create_examples_from_conll_format_file(os.path.join(data_dir, 'dev.txt'), 'dev')

    def get_test_examples(self, data_dir):
        return DataProcessor.create_examples_from_conll_format_file(os.path.join(data_dir, 'test.txt'), 'test')

    @staticmethod
    def get_labels():
        label_type = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC', '[CLS]', '[SEP]',
                      'Space']
        return label_type


class WeiboProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        return DataProcessor.create_examples_from_conll_format_file(os.path.join(data_dir, 'train.txt'), 'train')

    def get_dev_examples(self, data_dir):
        return DataProcessor.create_examples_from_conll_format_file(os.path.join(data_dir, 'dev.txt'), 'dev')

    def get_test_examples(self, data_dir):
        return DataProcessor.create_examples_from_conll_format_file(os.path.join(data_dir, 'test.txt'), 'test')

    @staticmethod
    def get_labels():
        label_type = ['O', 'B-PER.NAM', 'I-PER.NAM', 'B-PER.NOM', 'I-PER.NOM', 'B-ORG.NAM', 'I-ORG.NAM',
                      'B-ORG.NOM', 'I-ORG.NOM', 'B-LOC.NAM', 'I-LOC.NAM', 'B-LOC.NOM', 'I-LOC.NOM', 'B-GPE.NAM',
                      'I-GPE.NAM', 'B-GPE.NOM', 'I-GPE.NOM', '[CLS]', '[SEP]', 'Space']
        return label_type


class MSRAProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_examples(data_dir, 'train_content.txt', 'train_label.txt', 'train')

    def get_dev_examples(self, data_dir):
        return None

    def get_test_examples(self, data_dir):
        return self._create_examples(data_dir, 'testright_content.txt', 'testright_label.txt', 'test')

    @staticmethod
    def _create_examples(data_dir, content_file, label_file, set_type):
        examples = []
        index = 0
        for content_line, label_line in zip(codecs.open(os.path.join(data_dir, content_file), encoding='utf-8'),
                                            codecs.open(os.path.join(data_dir, label_file), encoding='utf-8')):
            guid = "%s-%d" % (set_type, index)
            words = content_line.split()
            labels = label_line.split()
            examples.append(InputExample(guid=guid, words=words, labels=labels))
            index += 1
        return examples

    @staticmethod
    def get_labels():
        label_type = ['O', 'B-NR', 'I-NR', 'B-NS', 'I-NS', 'B-NT', 'I-NT', '[CLS]', '[SEP]', 'Space']
        return label_type


class PD98Processor(MSRAProcessor):
    def get_train_examples(self, data_dir):
        return self._create_examples(data_dir, 'train_content.txt', 'train_label.txt', 'train')

    def get_dev_examples(self, data_dir):
        return self._create_examples(data_dir, 'dev_content.txt', 'dev_label.txt', 'train')

    def get_test_examples(self, data_dir):
        return None


def convert_examples_to_features(examples, max_seq_length, tokenizer, label_preprocessed, label_list):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    tokenize_info = []
    add_label = 'Space'
    for (ex_index, example) in enumerate(examples):
        tokenize_count = []
        tokens = ['[CLS]']
        predict_mask = [0]
        label_ids = [label_map['[CLS]']]
        for i, w in enumerate(example.words):
            sub_words = tokenizer.tokenize(w)
            if not sub_words:
                sub_words = ['[UNK]']
            tokenize_count.append(len(sub_words))
            tokens.extend(sub_words)
            if not label_preprocessed:
                for j in range(len(sub_words)):
                    if j == 0:
                        predict_mask.append(1)
                        label_ids.append(label_map[example.labels[i]])
                    else:
                        predict_mask.append(0)
                        label_ids.append(label_map[add_label])
        if label_preprocessed:
            predict_mask.extend([1] * len(example.labels))
            label_ids.extend([label_map[label] for label in example.labels])
            assert len(tokens) == len(label_ids), str(ex_index)
        tokenize_info.append(tokenize_count)

        if len(tokens) > max_seq_length - 1:
            logging.debug('Example {} is too long: {}'.format(ex_index, len(tokens)))
            tokens = tokens[0:(max_seq_length - 1)]
            predict_mask = predict_mask[0:(max_seq_length - 1)]
            label_ids = label_ids[0:(max_seq_length - 1)]
        tokens.append('[SEP]')
        predict_mask.append(0)
        label_ids.append(label_map['[SEP]'])

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        segment_ids = [0] * len(input_ids)
        input_mask = [1] * len(input_ids)

        # Pad up to the sequence length
        padding_length = max_seq_length - len(input_ids)
        zero_padding = [0] * padding_length
        input_ids += zero_padding
        input_mask += zero_padding
        segment_ids += zero_padding
        predict_mask += zero_padding
        label_ids += [label_map['Space']] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(predict_mask) == max_seq_length
        assert len(label_ids) == max_seq_length

        one_hot_labels = np.eye(len(label_list), dtype=np.float32)[label_ids]

        features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids,
                                      predict_mask=predict_mask, one_hot_labels=one_hot_labels))
    return features, tokenize_info


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x


def main(yaml_file):

    with open(yaml_file) as f:
        config = yaml.load(f.read())

    if not config['train']['do'] and not config['predict']['do']:
        raise ValueError("At least do training or do predicting in a run.")

    if config['use_cuda'] and torch.cuda.is_available():
        device = torch.device("cuda", torch.cuda.current_device())
        use_gpu = True
    else:
        device = torch.device("cpu")
        use_gpu = False
    logger.info("device: {}".format(device))

    processors = {
        "msra": MSRAProcessor,
        "weibo": WeiboProcessor,
        "pd98": PD98Processor,
        "conll": CONLLProcessor
    }

    task_name = config['task']['task_name'].lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % task_name)
    if task_name in ['msra', 'pd98']:
        label_preprocessed = True
    else:
        label_preprocessed = False
    processor = processors[task_name]()
    label_list = processor.get_labels()

    os.makedirs(config['task']['output_dir'], exist_ok=True)
    ckpts = [filename for filename in os.listdir(config['task']['output_dir']) if re.fullmatch('checkpoint-\d+', filename)]
    if config['task']['checkpoint'] or ckpts:
        if config['task']['checkpoint']:
            model_file = config['task']['checkpoint']
        else:
            model_file = os.path.join(config['task']['output_dir'], sorted(ckpts, key=lambda x: int(x[len('checkpoint-'):]))[-1])
        logging.info('Load %s' % model_file)
        checkpoint = torch.load(model_file, map_location='cpu')
        start_epoch = checkpoint['epoch']+1
        max_seq_length = checkpoint['max_seq_length']
        lower_case = checkpoint['lower_case']
        model = BertForNER.from_pretrained(config['task']['bert_model_dir'], state_dict=checkpoint['model_state'],
                                           num_labels=len(label_list))
    else:
        start_epoch = 0
        max_seq_length = config['task']['max_seq_length']
        lower_case = config['task']['lower_case']
        model = BertForNER.from_pretrained(config['task']['bert_model_dir'], num_labels=len(label_list))

    tokenizer = BertTokenizer.from_pretrained(config['task']['bert_model_dir'], do_lower_case=lower_case)

    model.to(device)

    if config['train']['do']:

        if config['train']['gradient_accumulation_steps'] < 1:
            raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                config['train']['gradient_accumulation_steps']))

        config['train']['batch_size'] = int(config['train']['batch_size'] / config['train']['gradient_accumulation_steps'])

        random.seed(config['train']['seed'])
        np.random.seed(config['train']['seed'])
        torch.manual_seed(config['train']['seed'])
        if use_gpu:
            torch.cuda.manual_seed_all(config['train']['seed'])

        train_examples = processor.get_train_examples(config['task']['data_dir'])
        num_train_steps = int(
            len(train_examples) / config['train']['batch_size'] / config['train']['gradient_accumulation_steps'] * config['train']['epochs'])

        # Prepare optimizer
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = BertAdam(optimizer_grouped_parameters, lr=config['train']['learning_rate'], warmup=config['train']['warmup_proportion'], t_total=num_train_steps)

        train_features, train_tokenize_info = convert_examples_to_features(train_examples, max_seq_length, tokenizer,
                                                                           label_preprocessed, label_list)
        with codecs.open(os.path.join(config['task']['output_dir'], "train.tokenize_info"), 'w', encoding='utf-8') as f:
            for item in train_tokenize_info:
                f.write(' '.join([str(num) for num in item])+'\n')
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", config['train']['batch_size'])
        logger.info("  Num steps = %d", num_train_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_predict_mask = torch.ByteTensor([f.predict_mask for f in train_features])
        all_one_hot_labels = torch.tensor([f.one_hot_labels for f in train_features], dtype=torch.float)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_predict_mask, all_one_hot_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=config['train']['batch_size'])

        model.train()
        global_step = int(
            len(train_examples) / config['train']['batch_size'] / config['train']['gradient_accumulation_steps'] * start_epoch)
        for epoch in trange(start_epoch, config['train']['epochs'], desc="Epoch"):
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, predict_mask, one_hot_labels = batch

                loss = model(input_ids, segment_ids, input_mask, predict_mask, one_hot_labels)

                if config['train']['gradient_accumulation_steps'] > 1:
                    loss = loss / config['train']['gradient_accumulation_steps']

                loss.backward()

                if (step + 1) % config['train']['gradient_accumulation_steps'] == 0:
                    # modify learning rate with special warm up BERT uses
                    lr_this_step = config['train']['learning_rate'] * warmup_linear(global_step/num_train_steps, config['train']['warmup_proportion'])
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

            # Save a checkpoint
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            torch.save({'epoch': epoch, 'model_state': model_to_save.state_dict(), 'max_seq_length': max_seq_length,
                        'lower_case': lower_case},
                       os.path.join(config['task']['output_dir'], 'checkpoint-%d' % epoch))

    if config['predict']['do']:
        if config['predict']['dataset'] == 'train':
            predict_examples = processor.get_train_examples(config['task']['data_dir'])
        elif config['predict']['dataset'] == 'dev':
            predict_examples = processor.get_dev_examples(config['task']['data_dir'])
        elif config['predict']['dataset'] == 'test':
            predict_examples = processor.get_test_examples(config['task']['data_dir'])
        else:
            raise ValueError("The dataset %s cannot be predicted." % config['predict']['dataset'])
        predict_features, predict_tokenize_info = convert_examples_to_features(predict_examples, max_seq_length,
                                                                               tokenizer, label_preprocessed, label_list)
        with codecs.open(os.path.join(config['task']['output_dir'], "test.tokenize_info"), 'w', encoding='utf-8') as f:
            for item in predict_tokenize_info:
                f.write(' '.join([str(num) for num in item])+'\n')
        logger.info("***** Running prediction *****")
        logger.info("  Num examples = %d", len(predict_examples))
        logger.info("  Batch size = %d", config['predict']['batch_size'])
        all_input_ids = torch.tensor([f.input_ids for f in predict_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in predict_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in predict_features], dtype=torch.long)
        predict_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
        # Run prediction for full data
        predict_sampler = SequentialSampler(predict_data)
        predict_dataloader = DataLoader(predict_data, sampler=predict_sampler, batch_size=config['predict']['batch_size'])
        model.eval()
        predictions = []
        for batch in predict_dataloader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids = batch
            logits = model(input_ids, segment_ids, input_mask)
            logits = logits.detach().cpu().numpy()
            predictions.extend(np.argmax(logits, -1).tolist())

        writer = codecs.open(os.path.join(config['task']['output_dir'], "%s.predict" % config['predict']['dataset']), 'w', encoding='utf-8')
        for predict_line, feature in zip(predictions, predict_features):
            predict_labels = []
            for index, label_id in enumerate(predict_line[:sum(feature.input_mask)]):
                if feature.predict_mask[index] == 1:
                    predict_labels.append(label_list[label_id])
            writer.write(' '.join(predict_labels)+'\n')
        writer.close()


if __name__ == "__main__":
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        main(sys.argv[1])
    else:
        print("Please specify the config file.")
