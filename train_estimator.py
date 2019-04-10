import os
import tensorflow as tf

from model_speech.cnn_ctc_estimator import AMEstimator
from utils import get_data, data_hparams


# 0.准备训练所需数据------------------------------
data_args = data_hparams()
data_args.data_type = 'train'
data_args.data_path = 'data/'
data_args.thchs30 = True
data_args.aishell = False
data_args.prime = False
data_args.stcmd = False
data_args.batch_size = 4
# data_args.data_length = 10
data_args.data_length = None
data_args.shuffle = True
train_data = get_data(data_args)


# 0.准备验证所需数据------------------------------
data_args = data_hparams()
data_args.data_type = 'dev'
data_args.data_path = 'data/'
data_args.thchs30 = True
data_args.aishell = False
data_args.prime = False
data_args.stcmd = False
data_args.batch_size = 4
data_args.data_length = None
# data_args.data_length = 10
data_args.shuffle = True
dev_data = get_data(data_args)

epochs = 10000
batch_size = 2
print(len(train_data.am_vocab))

# 1.声学模型训练-----------------------------------

feature_columns = []
feature_columns.append(tf.feature_column.numeric_column('the_inputs', shape=[200, 1], dtype=tf.float32))
# feature_columns.append(tf.feature_column.numeric_column('the_labels', shape=[1], dtype=tf.float32))
# feature_columns.append(tf.feature_column.numeric_column('input_length', shape=[], dtype=tf.int32))
# feature_columns.append(tf.feature_column.numeric_column('label_length', shape=[], dtype=tf.int32))



params= {
    'feature_columns': feature_columns
}
# model_dir = os.path.join()
am = AMEstimator(len(train_data.am_vocab), 'train', train_data.am_vocab, './logs_am_new_2', params, tf.estimator.RunConfig())

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


def data_input_fn(mode, tfrecord_path, batch_size, input_size):
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(lambda serial_exmp: parse_example(serial_exmp, input_size))
    # if mode == 'train':
    #     dataset = dataset.shuffle(buffer_size=1000, seed=123)
        # dataset = dataset.repeat(self._train_epochs)  # define outside loop
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(2 * batch_size)
    return dataset.make_one_shot_iterator().get_next()

# with tf.Session() as sess:
#     res = sess.run(data_input_fn('data/dev.tfrecord', batch_size))
#     print(res)

tf.logging.set_verbosity(tf.logging.INFO)
# train_spec = tf.estimator.TrainSpec(input_fn=lambda: data_input_fn('data/train2.tfrecord', batch_size))
# eval_spec = tf.estimator.EvalSpec(input_fn=lambda: data_input_fn('data/dev2.tfrecord', 1))
#
for i in range(epochs):
    tf.logging.info('############################## starting epoch {} #########################'.format(i+1))
    # tf.estimator.train_and_evaluate(am.ctc_model, train_spec, eval_spec)
    # tf.estimator.train_and_evaluate(am, train_spec, eval_spec)
    am.train(
        input_fn=lambda: data_input_fn('train','data/train3.tfrecord', batch_size, 1568*200),
        hooks=None,
        steps=None,
        max_steps=None,
        saving_listeners=None)

    result = am.evaluate(
        input_fn=lambda: data_input_fn('eval','data/dev3.tfrecord', batch_size, 1480*200),
        steps=None,  # Number of steps for which to evaluate model.
        hooks=None,
        checkpoint_path=None,  # latest checkpoint in model_dir is used.
        name=None)

    print(result)

    # am.ctc_model.evaluate(
    #     input_fn=lambda: data_input_fn('data/dev2.tfrecord', batch_size),
    #     steps=None,  # Number of steps for which to evaluate model.
    #     hooks=None,
    #     checkpoint_path=None,  # latest checkpoint in model_dir is used.
    #     name=None)



