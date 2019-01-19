import os
import codecs
import logging
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from .pytorch_pretrained_bert.tokenization import BertTokenizer
from .pytorch_pretrained_bert.modeling import PreTrainedBertModel, BertModel

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class BertForNER(PreTrainedBertModel):

    def __init__(self, config, num_labels):
        super(BertForNER, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.hidden2label = torch.nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, segment_ids, input_mask, predict_mask=None, one_hot_labels=None):
        bert_layer, _ = self.bert(input_ids, segment_ids, input_mask, output_all_encoded_layers=False)
        logits = self.hidden2label(bert_layer)

        if one_hot_labels is not None:
            p = torch.nn.functional.log_softmax(logits, -1)
            losses = -torch.sum(one_hot_labels*p, -1)
            losses = torch.masked_select(losses, predict_mask)
            return torch.sum(losses)
        else:
            return logits


class InputFeatures(object):

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


class NJUNER:

    def __init__(self, model_dir, batch_size=8, no_cuda=False):
        if not os.path.exists(model_dir):
            logging.error("%s is not a valid path." % model_dir)
            exit(1)
        if len(set(['bert_config.json', 'checkpoint', 'vocab.txt']) - set(os.listdir(model_dir))) > 0:
            logging.error('%s not a valid model directory', model_dir)
            exit(1)
        checkpoint = torch.load(os.path.join(model_dir, 'checkpoint'), map_location='cpu')
        self._max_seq_length = checkpoint['max_seq_length']
        self._label_list = ['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', '[CLS]', '[SEP]', 'Space']
        logger.info("Loading the model")
        self._tokenizer = BertTokenizer.from_pretrained(model_dir, do_lower_case=checkpoint['lower_case'])
        self._model = BertForNER.from_pretrained(model_dir, state_dict=checkpoint['model_state'],
                                                 num_labels=len(self._label_list))
        self._batch_size = batch_size
        if not no_cuda and torch.cuda.is_available():
            self._device = torch.device("cuda", torch.cuda.current_device())
        else:
            self._device = torch.device("cpu")
        self._model.to(self._device)
        logger.info("The model is ready")

    def _split_paragraph(self, ori_lines):
        lines = []
        for index, line in enumerate(ori_lines):
            if len(line) > self._max_seq_length - 2:
                last_index = 0
                for s_index, c in enumerate(line):
                    if c == '。' or c == '.':
                        lines.append(line[last_index:s_index + 1])
                        last_index = s_index + 1
                if last_index < len(line):
                    lines.append(line[last_index:])
            else:
                lines.append(line)
        return lines

    def _get_features(self, ori_lines):
        lines = self._split_paragraph(ori_lines)
        features = []
        tokens_list = []
        for index, line in enumerate(lines):
            tokens = ['[CLS]']
            chars = self._tokenizer.tokenize(line)
            tokens.extend(chars)
            if len(tokens) > self._max_seq_length - 1:
                logging.debug('Example {} is too long: {}'.format(index, line))
                tokens = tokens[0:(self._max_seq_length - 1)]
            tokens.append('[SEP]')
            input_ids = self._tokenizer.convert_tokens_to_ids(tokens)
            segment_ids = [0] * len(input_ids)
            input_mask = [1] * len(input_ids)
            zero_padding = [0] * (self._max_seq_length - len(input_ids))
            input_ids += zero_padding
            input_mask += zero_padding
            segment_ids += zero_padding
            features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids))
            tokens_list.append(tokens[1:-1])
        return features, tokens_list

    def _predict_features(self, features, tokens):
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        predict_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
        predict_sampler = SequentialSampler(predict_data)
        predict_dataloader = DataLoader(predict_data, sampler=predict_sampler, batch_size=self._batch_size)
        self._model.eval()
        predict_ids = []
        for batch in predict_dataloader:
            batch = tuple(t.to(self._device) for t in batch)
            input_ids, input_mask, segment_ids = batch
            logits = self._model(input_ids, segment_ids, input_mask)
            logits = logits.detach().cpu().numpy()
            predict_ids.extend(np.argmax(logits, -1).tolist())
        predictions = []
        for token_line, predict_line in zip(tokens, predict_ids):
            predictions.append([self._label_list[label_id] for label_id in predict_line[1: 1+len(token_line)]])
        return predictions

    def label(self, sentences):
        features, tokens = self._get_features(sentences)
        predictions = self._predict_features(features, tokens)
        ret = []
        for sen_predictions, sen_tokens in zip(predictions, tokens):
            ret.append([(tag, token) for tag, token in zip(sen_predictions, sen_tokens)])
        return ret

    def predict_file(self, input_file_path, output_dir, conll_format):
        if not conll_format:
            sentences = []
            for line in codecs.open(input_file_path, 'r', encoding='utf-8'):
                sentences.append(line.strip('\n'))
            features, tokens = self._get_features(sentences)
        else:
            sentences = []
            words = []
            for line in codecs.open(input_file_path, 'r', encoding='utf-8'):
                if not line.strip():
                    sentences.append(words)
                    words = []
                else:
                    segs = line.split()
                    words.append(segs[0])
            features, tokens = [], []
            for index, line in enumerate(sentences):
                line_tokens = ['[CLS]']
                for w in line:
                    chars = self._tokenizer.tokenize(w)
                    if not chars:
                        chars = ['[UNK]']
                    line_tokens.extend(chars)
                if len(line_tokens) > self._max_seq_length - 1:
                    logging.debug('Example {} is too long: {}'.format(index, line))
                    line_tokens = line_tokens[0:(self._max_seq_length - 1)]
                line_tokens.append('[SEP]')
                input_ids = self._tokenizer.convert_tokens_to_ids(line_tokens)
                segment_ids = [0] * len(input_ids)
                input_mask = [1] * len(input_ids)
                zero_padding = [0] * (self._max_seq_length - len(input_ids))
                input_ids += zero_padding
                input_mask += zero_padding
                segment_ids += zero_padding
                features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids))
                tokens.append(line_tokens[1:-1])
        predictions = self._predict_features(features, tokens)
        tokens_writer = codecs.open(os.path.join(output_dir, 'tokens.txt'), 'w', encoding='utf-8')
        predict_writer = codecs.open(os.path.join(output_dir, 'predictions.txt'), 'w', encoding='utf-8')
        summary_writer = codecs.open(os.path.join(output_dir, 'summary.txt'), 'w', encoding='utf-8')
        for token_line, predict_line in zip(tokens, predictions):
            tokens_writer.write(' '.join(token_line) + '\n')
            predict_writer.write(' '.join(predict_line) + '\n')
            spans = bies_tags_to_spans(predict_line)
            summary_writer.write(
                str([(item[0], ''.join(token_line[item[1][0]: item[1][1] + 1])) for item in spans]) + '\n')
        tokens_writer.close()
        predict_writer.close()
        summary_writer.close()


def bies_tags_to_spans(tag_sequence):
    spans = list()
    span_start = 0
    span_end = 0
    active_tag = None
    for index, string_tag in enumerate(tag_sequence):
        bies_tag = string_tag[0]
        if bies_tag not in ["B", "I", "E", "S", "O"]:
            continue
        if string_tag[2:] != "O":
            entity_tag = string_tag[2:]
        else:
            entity_tag = None
        if bies_tag == "O":
            if active_tag is not None:
                spans.append((active_tag, (span_start, span_end)))
            active_tag = None
        elif bies_tag == "B" or bies_tag == "S":
            if active_tag is not None:
                spans.append((active_tag, (span_start, span_end)))
            active_tag = entity_tag
            span_start = index
            span_end = index
        elif bies_tag == "I":
            if active_tag is not None:
                if entity_tag == active_tag:
                    span_end += 1
                else:
                    spans.append((active_tag, (span_start, span_end)))
                    active_tag = entity_tag
                    span_start = index
                    span_end = index
            else:
                active_tag = entity_tag
                span_start = index
                span_end = index
        elif bies_tag == "E":
            if active_tag is not None:
                if entity_tag == active_tag:
                    spans.append((active_tag, (span_start, span_end + 1)))
                else:
                    spans.append((active_tag, (span_start, span_end)))
            active_tag = None
    if active_tag is not None:
        spans.append((active_tag, (span_start, span_end)))

    return spans
