import re
import codecs
from pytorch_pretrained_bert.tokenization import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('ch_model', do_lower_case=True)


def preprocess(path, dataset):
    label_type = ['O', 'NR', 'NS', 'NT']
    end = [';', '?', '!', '；', '。', '？', '！']
    max_length = 178
    out_content = codecs.open('{}/{}_content.txt'.format(path, dataset), 'w', encoding='utf-8')
    out_label = codecs.open('{}/{}_label.txt'.format(path, dataset), 'w', encoding='utf-8')
    out_line_count = codecs.open('{}/{}_line_count.txt'.format(path, dataset), 'w', encoding='utf-8')
    index = 0
    all_count = 0
    for line in codecs.open('{}/{}.txt'.format(path, dataset), encoding='utf-8'):
        words, labels = [], []
        for item in re.split(' +', line.strip('\n')):
            if item:
                w = item.split('/')[0]
                l = item.split('/')[1].upper()
                if not w or l not in label_type:
                    print('{}: {}'.format(str(index), item))
                    continue
                words.append(w)
                labels.append(l)
        assert len(words) == len(labels)

        result = []
        content_line, label_line = [], []
        i = 0
        while i < len(words):
            content_line.append(words[i])
            if words[i] in end:
                label_line.append('O')
                if i + 1 < len(words) and (words[i + 1] == '"' or words[i + 1] == '”'):
                    content_line.append(words[i + 1])
                    label_line.append('O')
                    i += 1
                result.append((content_line, label_line))
                content_line, label_line = [], []
            else:
                chars = tokenizer.tokenize(words[i])
                if labels[i] == 'O':
                    start_l = 'O'
                    middle_l = 'O'
                else:
                    start_l = 'B-'+labels[i]
                    middle_l = 'I-'+labels[i]
                label_line.append(start_l)
                if len(chars) > 1:
                    for _ in chars[1:]:
                        label_line.append(middle_l)
            i += 1
        if content_line:
            result.append((content_line, label_line))

        count = 0
        for item in result:
            if len(item[1]) > max_length:
                temp_line = ''.join(item[0])
                comma_index = [i for i, x in enumerate(temp_line) if x == '，']
                if comma_index:
                    last_indx1, last_indx2 = 0, 0
                    for indx in comma_index:
                        seg = temp_line[last_indx1: indx+1]
                        out_content.write(seg+'\n')
                        last_indx1 = indx + 1
                        seg_label_len = len(tokenizer.tokenize(seg))
                        out_label.write(' '.join(item[1][last_indx2: last_indx2+seg_label_len])+'\n')
                        last_indx2 += seg_label_len
                        count += 1
                        all_count += 1
                else:
                    print(str(index) + ' ' + str(all_count))
                    out_content.write(' '.join(item[0]) + '\n')
                    out_label.write(' '.join(item[1]) + '\n')
                    count += 1
                    all_count += 1
            else:
                out_content.write(' '.join(item[0]) + '\n')
                out_label.write(' '.join(item[1]) + '\n')
                count += 1
                all_count += 1
        out_line_count.write(str(index) + ' ' + str(count) + '\n')
        index += 1
