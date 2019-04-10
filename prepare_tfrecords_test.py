import os
import difflib
import numpy as np
import tensorflow as tf
import scipy.io.wavfile as wav
from tqdm import tqdm
from scipy.fftpack import fft
from python_speech_features import mfcc
from random import shuffle
# from keras import backend as K

# output file name string to a queue
filename_queue = tf.train.string_input_producer(['data/train3.tfrecord'], num_epochs=None)
# create a reader from file queue
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
# get feature from serialized example

features = tf.parse_single_example(serialized_example,
                                   features={
                                            'the_inputs':tf.FixedLenFeature([1568*200], tf.float32),
                                             'the_labels':tf.VarLenFeature(tf.float32),
                                             'input_length':tf.FixedLenFeature([1], tf.int64),
                                             # 'label_length':tf.FixedLenFeature([1], tf.int64)
                                             })

# the_inputs, the_labels, input_length, label_length = tf.train.shuffle_batch([tf.reshape(tf.decode_raw(features['the_inputs'], tf.float32), shape=[-1, 200, 1]), features['the_labels'], features['input_length'], features['label_length']], batch_size=1,
#                                                    capacity=200, min_after_dequeue=100, num_threads=2)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

tf.train.start_queue_runners(sess=sess)
input, label, input_l = sess.run([
                                            tf.reshape(features['the_inputs'], shape=[-1,1568,200,1]),
                                           features['the_labels'],
                                           features['input_length'],
                                           # features['label_length']
                                           ])
# print(a_val, b_val, c_val)
print('first batch:')
print('  input:',input.shape)
print('  label:',label)
print('  input_l:',input_l)
# print('  label_l:',label_l)


