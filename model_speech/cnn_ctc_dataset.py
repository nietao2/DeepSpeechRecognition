import tensorflow as tf
from tqdm import tqdm
import numpy as np
import scipy.io.wavfile as wav
from scipy.fftpack import fft


class AudioDataset():
    def __init__(self, wav_lst, pny_lst, label_vocab, batch_size):
        self.pny_lst = pny_lst
        self.wav_lst = wav_lst
        self.label_vocab = label_vocab
        self.batch_size = batch_size

    def pny2id(self, line, vocab):
        return [vocab.index(pny) for pny in line]

    def compute_fbank(self, file):
        x = np.linspace(0, 400 - 1, 400, dtype=np.int64)
        w = 0.54 - 0.46 * np.cos(2 * np.pi * (x) / (400 - 1))  # 汉明窗
        fs, wavsignal = wav.read(file)
        # wav波形 加时间窗以及时移10ms
        time_window = 25  # 单位ms
        wav_arr = np.array(wavsignal)
        range0_end = int(len(wavsignal) / fs * 1000 - time_window) // 10  # 计算循环终止的位置，也就是最终生成的窗数
        data_input = np.zeros((range0_end, 200), dtype=np.float)  # 用于存放最终的频率特征数据
        data_line = np.zeros((1, 400), dtype=np.float)
        for i in range(0, range0_end):
            p_start = i * 160
            p_end = p_start + 400
            data_line = wav_arr[p_start:p_end]
            data_line = data_line * w  # 加窗
            data_line = np.abs(fft(data_line))
            data_input[i] = data_line[0:200]  # 设置为400除以2的值（即200）是取一半数据，因为是对称的
        data_input = np.log(data_input + 1)
        # data_input = data_input[::]
        return data_input

    def ctc_len(self, label):
        add_len = 0
        label_len = len(label)
        for i in range(label_len - 1):
            if label[i] == label[i + 1]:
                add_len += 1
        return label_len + add_len

    def _generator_fn(self):
        for index in range(len(self.wav_lst)):
            fbank = self.compute_fbank(self.wav_lst[index])
            pad_fbank = np.zeros((fbank.shape[0] // 8 * 8 + 8, fbank.shape[1]))
            pad_fbank[:fbank.shape[0], :] = fbank
            label = self.pny2id(self.pny_lst[index], self.label_vocab)
            label_ctc_len = self.ctc_len(label)
            if pad_fbank.shape[0] // 8 >= label_ctc_len:
                inputs = {'the_inputs': pad_fbank,
                          'the_labels': label,
                          'input_length': len(pad_fbank) // 8,
                          'label_length': len(label),
                          }
                yield inputs

    def _map_func(self, input):
        input['the_labels'] = tf.contrib.layers.dense_to_sparse(input['the_labels'])
        # inputs['the_inputs'] = tf.contrib.layers.dense_to_sparse(inputs['the_inputs'])
        return input

    def _batch_fn(self, input):
        return tf.data.Dataset.zip({'the_inputs':input['the_inputs'].padded_batch(self.batch_size, padded_shapes=[None, 200]),
                                    'the_labels':input['the_labels'].batch(self.batch_size),
                                    'input_length':input['input_length'].batch(self.batch_size)})

    def _input_fn(self, mode, shuffle=True):
        output_types = {
            'the_inputs': tf.float32,
            'the_labels': tf.int32,
            'input_length': tf.int32,
            'label_length': tf.int32
        }
        output_shapes = {'the_inputs': [None, 200],'the_labels': [None], 'input_length': [], 'label_length': []}
        dataset = tf.data.Dataset.from_generator(self._generator_fn,
                                                 output_types = output_types,
                                                 output_shapes=output_shapes,
                                                 args=None)
        dataset = dataset.map(self._map_func, num_parallel_calls=8)
        if mode == 'train' and shuffle: # for training
            dataset = dataset.shuffle(128*self.batch_size)
        # dataset = dataset.padded_batch(batch_size,
        #                                padded_shapes={'the_inputs': [None, 200], 'input_length': [], 'label_length': []},
        #                                # padding_values={'the_inputs': 0.0,'the_labels': 0, 'input_length': 0, 'label_length': 0}
        #                                )
        # dataset = dataset.batch(batch_size)
        dataset = dataset.window(self.batch_size)
        dataset = dataset.flat_map(self._batch_fn)
        dataset = dataset.prefetch(2 * self.batch_size)
        return dataset.make_one_shot_iterator().get_next()


def input_fn(mode, batch_size, wav_lst, pny_lst, label_vocab, shuffle=True):
    dataset = AudioDataset(wav_lst, pny_lst, label_vocab, batch_size)
    return dataset._input_fn(mode, shuffle)


def load_data(file_paths, data_path, size=None):
    pny_lst = []
    wav_lst = []
    for file in file_paths:
        with open(file, mode='r', encoding='utf-8') as f:
            data = f.readlines()
            for line in tqdm(data):
                wav, pny, han = line.split('\t')
                pny_lst.append(pny.split(' '))
                wav_lst.append(data_path + wav)
    if size is not None:
        wav_lst = wav_lst[:size]
        pny_lst = pny_lst[:size]
    return wav_lst, pny_lst


def load_vocab(file_paths):
    label_vocab = ['<pad>']
    for file in tqdm(file_paths):
        with open(file, mode='r', encoding='utf-8') as f:
            data = f.readlines()
            for line in tqdm(data):
                _, pny, _ = line.split('\t')
                for term in pny.split(' '):
                    if term not in label_vocab:
                        label_vocab.append(term)
    # label_vocab.append('_')
    return label_vocab
