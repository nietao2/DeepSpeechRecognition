import unittest

import tensorflow as tf

from model_language.transformer_dataset import input_fn, load_vocab, load_data


class TransformerDatasetTest(unittest.TestCase):

    def test(self):
        with tf.Session() as sess:
            input_vocab, label_vocab = load_vocab(['../data/thchs_train.txt', '../data/thchs_dev.txt', '../data/thchs_test.txt'])
            pny_lst, han_lst = load_data(['../data/thchs_train.txt'])
            records = sess.run(input_fn('train', 4, 50, pny_lst, han_lst, input_vocab, label_vocab))
            print(records)