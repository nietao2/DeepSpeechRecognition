#coding=utf-8
import os
import difflib
import tensorflow as tf
import numpy as np

from model_speech.cnn_ctc_estimator import AMEstimator
from utils import decode_ctc, GetEditDistance


# 0.准备解码所需字典，参数需和训练一致，也可以将字典保存到本地，直接进行读取
from utils import get_data, data_hparams
data_args = data_hparams()
data_args.thchs30 = True
data_args.aishell = False
data_args.prime = False
data_args.stcmd = False
# data_args.data_length = None
train_data = get_data(data_args)


# 1.声学模型-----------------------------------
am = AMEstimator(len(train_data.am_vocab), 'infer', train_data.am_vocab, '/Users/nietao/IdeaProjects/DeepSpeechRecognition/logs_am_new_2', None, tf.estimator.RunConfig())

# # 3. 准备测试所需数据， 不必和训练数据一致，通过设置data_args.data_type测试，
# #    此处应设为'test'，我用了'train'因为演示模型较小，如果使用'test'看不出效果，
# #    且会出现未出现的词。
# data_args.data_type = 'train'
# data_args.shuffle = False
# data_args.batch_size = 1
# test_data = get_data(data_args)
#
# # 4. 进行测试-------------------------------------------
# am_batch = test_data.get_am_batch()
# word_num = 0
# word_error_num = 0


def parse_example(serial_exmp, input_size):
    feats = tf.parse_single_example(serial_exmp,
                                    features={'the_inputs':tf.FixedLenFeature([input_size], tf.float32), #1568*200      1480*200  233600
                                              'the_labels':tf.VarLenFeature(tf.float32),   #47
                                              'input_length':tf.FixedLenFeature([1], tf.int64),
                                              # 'label_length':tf.FixedLenFeature([1], tf.int64)
                                              })
    feats['the_inputs'] = tf.reshape(feats['the_inputs'], shape=[-1, 200, 1])
    # feats['the_labels'] = tf.reshape(feats['the_labels'], shape=[45])
    # feats['input_length'] = tf.reshape(tf.cast(feats['input_length'], tf.int32), shape=[])
    # feats['label_length'] = tf.reshape(tf.cast(feats['label_length'], tf.int32), shape=[])
    # label = feats.pop('the_labels')
    label = tf.zeros(name='ctc', shape=(), dtype=tf.int32)

    return feats, label


def data_input_fn(tfrecord_path, batch_size, input_size):
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(lambda serial_exmp: parse_example(serial_exmp, input_size))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(2 * batch_size)
    return dataset.make_one_shot_iterator().get_next()

# with tf.Session() as sess:
#     res = sess.run(data_input_fn('data/train3.tfrecord', 1))
#     print(res)

# dataset = tf.data.Dataset.from_tensor_slices((feats, label))
# input_fn = dataset.batch(1).make_one_shot_iterator().get_next()
#
result = am.predict(input_fn = lambda: data_input_fn('data/train3.tfrecord', 1, 1568*200),
                    predict_keys=None,
                    hooks=None,
                    checkpoint_path=None,
                    yield_single_examples=True)
print(result)
#
for r in result:
    text = []
    for i in r['text_ids']:
        text.append(train_data.am_vocab[i])
    text = ' '.join(text)
    print('文本结果：', text)
    text = []
    for i in r['y_true']:
        text.append(train_data.am_vocab[i])
    text = ' '.join(text)
    print('原文结果：', text)
# print('原文结果：', ' '.join(feats['the_labels']))