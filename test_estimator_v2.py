#coding=utf-8
import sys

import tensorflow as tf

from model_speech.cnn_ctc_dataset import input_fn, load_vocab, load_data
from model_speech.cnn_ctc_estimator_v2 import AMEstimator

label_vocab = load_vocab(['./data/thchs_train.txt', './data/thchs_dev.txt', './data/thchs_test.txt'])
wav_lst, pny_lst = load_data(['./data/thchs_train.txt'], './data/', size=4)
# dev_wav_lst, dev_pny_lst = load_data(['./data/thchs_dev.txt'], './data/')
config = tf.ConfigProto()
config.intra_op_parallelism_threads = 8
config.inter_op_parallelism_threads = 8
run_config = tf.estimator.RunConfig().replace(
    session_config=config)
am = AMEstimator(len(label_vocab), 'train', label_vocab, './logs_am_new_3', None, run_config)
result = am.predict(input_fn = lambda: input_fn('pred', 4, wav_lst, pny_lst, label_vocab),
                    predict_keys=None,
                    hooks=None,
                    checkpoint_path=None,
                    yield_single_examples=True)
print(result)
#
for r in result:
    text = []
    print(r['input_length'])
    # print(r['label_length'])
    for i in r['text_ids']:
        text.append(label_vocab[i])
    text = ' '.join(text)
    print('文本结果：', text)
    text = []
    for i in r['y_true']:
        text.append(label_vocab[i])
    text = ' '.join(text)
    print('原文结果：', text)
# print('原文结果：', ' '.join(feats['the_labels']))