import codecs
import random
import sys


def prepare_dataset():
    label_type = ['nr', 'ns', 'nt']
    w1 = codecs.open('pd98/train.txt', 'w', encoding='utf-8')
    w2 = codecs.open('pd98/dev.txt', 'w', encoding='utf-8')
    dev_count = 1948
    for index, line in enumerate(codecs.open('pd98/199801.txt', 'r', encoding='utf-8')):
        segs = line.strip('\n').split('\t')
        ner_list = []
        i = 0
        while i < len(segs):
            if segs[i][0] == '[':
                for j in range(i+1, len(segs)):
                    if ']' in segs[j]:
                        label = segs[j].split(']/')[1]
                        if label not in label_type:
                            label = 'o'
                        break
                assert j < len(segs), str(index)
                chunk = segs[i].split('/')[0][1:]
                for k in range(i+1, j+1):
                    chunk += segs[k].split('/')[0]
                ner_list.append((chunk, label))
                i = j+1
            else:
                word, label = segs[i].split('/')
                if label not in label_type:
                    label = 'o'
                ner_list.append((word, label))
                i += 1
        if dev_count > 0 and random.random() > 0.8:
            w2.write(' '.join([word+'/'+label for word, label in ner_list])+'\n')
            dev_count -= 1
        else:
            w1.write(' '.join([word + '/' + label for word, label in ner_list]) + '\n')
    w1.close()
    w2.close()


if __name__ == '__main__':
    if sys.argv[1] == '0':
        prepare_dataset()
    else:
        import preprocess_msra
        preprocess_msra.preprocess('pd98', 'train')
        preprocess_msra.preprocess('pd98', 'dev')
