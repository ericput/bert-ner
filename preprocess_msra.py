import re
import codecs
from pytorch_pretrained_bert.tokenization import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-pretrained/chinese_L-12_H-768_A-12', do_lower_case=True)


def preprocess(path, dataset):
    label_type = ['O', 'NR', 'NS', 'NT']
    end = [';', '?', '!', '；', '。', '？', '！']
    comma = '，'
    max_length = 178
    out_line_count = codecs.open('{}/{}_line_count.txt'.format(path, dataset), 'w', encoding='utf-8')
    sens = []
    for index, line in enumerate(codecs.open('{}/{}.txt'.format(path, dataset), encoding='utf-8')):
        words, labels = [], []
        for item in re.split(' +', line.strip()):
            if item:
                w = item.split('/')[0]
                l = item.split('/')[1].upper()
                if not w or l not in label_type:
                    print('{}: {}'.format(str(index), item))
                    continue
                words.append(w)
                labels.append(l)
        assert len(words) == len(labels)

        if len(''.join(words)) > (max_length - 20):
            content_line, label_line, _split_result = [], [], []
            i = 0
            while i < len(words):
                content_line.append(words[i])
                label_line.append(labels[i])
                if words[i] in end:
                    if i + 1 < len(words) and (words[i + 1] == '"' or words[i + 1] == '”'):
                        content_line.append(words[i + 1])
                        label_line.append(labels[i + 1])
                        i += 1
                    _split_result.append((content_line, label_line))
                    content_line, label_line = [], []
                i += 1
            if content_line:
                _split_result.append((content_line, label_line))

            split_result = []
            for pair in _split_result:
                if len(''.join(pair[0])) > (max_length - 20):
                    comma_index = [i for i, w in enumerate(pair[0]) if comma in w]
                    last_indx = 0
                    for indx in comma_index:
                        split_result.append((pair[0][last_indx: indx+1], pair[1][last_indx: indx+1]))
                        last_indx = indx + 1
                    if last_indx < len(pair[0]):
                        split_result.append((pair[0][last_indx:], pair[1][last_indx:]))
                else:
                    split_result.append(pair)
        else:
            split_result = [(words, labels)]

        out_line_count.write(str(len(split_result))+'\n')

        sens.extend(split_result)

    out_line_count.close()

    out_content = codecs.open('{}/{}_content.txt'.format(path, dataset), 'w', encoding='utf-8')
    out_label = codecs.open('{}/{}_label.txt'.format(path, dataset), 'w', encoding='utf-8')
    for index, item in enumerate(sens):
        content_line, label_line = item
        units, labels = [], []
        for word, label in zip(content_line, label_line):
            _units = tokenizer.tokenize(word)
            units.extend(_units)
            if label == 'O':
                start_l = 'O'
                middle_l = 'O'
            else:
                start_l = 'B-' + label
                middle_l = 'I-' + label
            labels.append(start_l)
            if len(_units) > 1:
                for _ in _units[1:]:
                    labels.append(middle_l)
        assert len(units) == len(labels), str(index)
        if len(units) > max_length:
            print(index)
        out_content.write(' '.join(units)+'\n')
        out_label.write(' '.join(labels)+'\n')
    out_content.close()
    out_label.close()
