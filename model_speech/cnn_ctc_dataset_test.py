import unittest

import tensorflow as tf

from model_speech.cnn_ctc_dataset import input_fn, load_vocab, load_data


class TransformerDatasetTest(unittest.TestCase):

    def test(self):
        with tf.Session() as sess:
            label_vocab = load_vocab(['../data/thchs_train.txt', '../data/thchs_dev.txt', '../data/thchs_test.txt'])
            wav_lst, pny_lst = load_data(['../data/thchs_train.txt'], '../data/')
            records = sess.run(input_fn('train', 1, wav_lst, pny_lst, label_vocab, shuffle=False))
            print(records[0]['the_inputs'])
            import json
            with open('../result.json', 'w') as fp:
                json.dump(records[0]['the_inputs'].tolist(), fp)
