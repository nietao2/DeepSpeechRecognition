import sys

import tensorflow as tf

from model_speech.cnn_ctc_dataset import input_fn, load_vocab, load_data
from model_speech.cnn_ctc_estimator_v2 import AMEstimator


def train():
    label_vocab = load_vocab(['./data/thchs_train.txt', './data/thchs_dev.txt', './data/thchs_test.txt'])
    wav_lst, pny_lst = load_data(['./data/thchs_train.txt'], './data/', size=1)
    dev_wav_lst, dev_pny_lst = load_data(['./data/thchs_dev.txt'], './data/')
    config = tf.ConfigProto()
    config.intra_op_parallelism_threads = 8
    config.inter_op_parallelism_threads = 8
    run_config = tf.estimator.RunConfig().replace(
        session_config=config)
    am = AMEstimator(len(label_vocab), 'train', label_vocab, './logs_am_new_3', None, run_config)

    epochs = 1000
    label_error_rate = 1000
    for i in range(epochs):
        tf.logging.info('############################## starting epoch {} #########################'.format(i+1))

        am.train(input_fn=lambda: input_fn('train', 4, wav_lst, pny_lst, label_vocab, shuffle=True),
                 hooks=None,
                 steps=None,
                 max_steps=None,
                 saving_listeners=None)
        save_model(am)
        # result = am.evaluate(input_fn=lambda: input_fn('eval', 4, dev_wav_lst, dev_pny_lst, label_vocab, shuffle=False),
        #             steps=None,  # Number of steps for which to evaluate model.
        #             hooks=None,
        #             checkpoint_path=None,  # latest checkpoint in model_dir is used.
        #             name=None)
        #
        # if label_error_rate > result['label_error_rate']:
        #     label_error_rate = result['label_error_rate']
        #     save_model(am)

def save_model(model):
    feature_spec = {
        'the_inputs':tf.FixedLenSequenceFeature([200], tf.float32, allow_missing=True),
         # 'the_labels':tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
         'input_length':tf.FixedLenFeature([], tf.int64, default_value=0),
    }
    serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
    result = model.export_savedmodel('export', serving_input_receiver_fn)
    print(result)

def main(argv):
    train()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main, argv=[sys.argv[0]])
