import tensorflow as tf
from tqdm import tqdm


class LMDataset():

    def __init__(self, input_max_len, label_max_len, pny_lst, han_lst, input_vocab, label_vocab):
        self.input_max_len = input_max_len
        self.label_max_len = label_max_len
        self.pny_lst = pny_lst
        self.han_lst = han_lst
        self.input_vocab = input_vocab
        self.label_vocab = label_vocab

    def _generator_fn(self, mode):
        for input, label in zip(self.pny_lst, self.han_lst):
            x = self._encode(input, self.input_vocab)
            label = ''.join(label.split(' '))
            if mode == 'pred':
                y = [0] * len(label) + [0]
            else:
                y = self._encode(label, self.label_vocab)

            x = x + [0] * (self.input_max_len - len(x) - 1)

            y = y + [0] * (self.label_max_len - len(y) - 1)
            inputs = {'x': x,
                      'y': y,
                      }
            yield inputs

    def _encode(self, list, vocab):
        return [vocab.index(term) for term in list] + [len(vocab)-1]

    def _input_fn(self, mode, batch_size, shuffle=True):
        # output_type = ((tf.int32), (tf.int32))
        # output_shapes = ([None], [None])
        output_shapes = ({'x': [None], 'y': [None]})
        output_types = ({
                            'x': tf.int32,
                            'y': tf.int32,
                        })
        dataset = tf.data.Dataset.from_generator(self._generator_fn,
                                                 output_types = output_types,
                                                 output_shapes=output_shapes,
                                                 args=[mode])
        if mode == 'train' and shuffle: # for training
            dataset = dataset.shuffle(128*batch_size)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(2 * batch_size)
        return dataset.make_one_shot_iterator().get_next()


def input_fn(mode, batch_size, input_max_len, label_max_len, pny_lst, han_lst, input_vocab, label_vocab):
    dataset = LMDataset(input_max_len, label_max_len, pny_lst, han_lst, input_vocab, label_vocab)
    return dataset._input_fn(mode, batch_size)


def load_data(file_paths, size=None):
    pny_lst = []
    han_lst = []
    for file in file_paths:
        with open(file, mode='r', encoding='utf-8') as f:
            data = f.readlines()
            for line in tqdm(data):
                _, pny, han = line.split('\t')
                pny_lst.append(pny.split(' '))
                han_lst.append(han.strip('\n'))
    if size is not None:
        pny_lst = pny_lst[:size]
        han_lst = han_lst[:size]
    return pny_lst, han_lst


def load_vocab(file_paths):
    input_vocab = ['<pad>']
    label_vocab = ['<pad>']
    for file in tqdm(file_paths):
        with open(file, mode='r', encoding='utf-8') as f:
            data = f.readlines()
            for line in tqdm(data):
                _, pny, han = line.split('\t')
                for term in pny.split(' '):
                    if term not in input_vocab:
                        input_vocab.append(term)
                han = ''.join(han.strip('\n').split(' '))
                for term in han:
                    if term not in label_vocab:
                        label_vocab.append(term)
    input_vocab.append('<end>')
    label_vocab.append('<end>')
    return input_vocab, label_vocab
