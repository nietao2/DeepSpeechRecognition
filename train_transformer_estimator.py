import sys

import tensorflow as tf

from model_language.transformer_dataset import input_fn, load_vocab, load_data
from model_language.transformer_estimator import LMTransformer, transformer_hparams


def train():
    input_vocab, label_vocab = load_vocab(
        ['./data/thchs_train.txt', './data/thchs_dev.txt', './data/thchs_test.txt'])
    pny_lst, han_lst = load_data(['./data/thchs_train.txt'])
    dev_pny_lst, dev_han_lst = load_data(['./data/thchs_dev.txt'])
    hp = transformer_hparams()
    hp.input_vocab_size = len(input_vocab)
    hp.label_vocab_size = len(label_vocab)
    config = tf.ConfigProto()
    config.intra_op_parallelism_threads = 8
    config.inter_op_parallelism_threads = 8
    run_config = tf.estimator.RunConfig().replace(
        session_config=config)
    lm = LMTransformer(hp, model_dir='logs_lm_new', params=None, config=run_config)

    epochs = 1000
    for i in range(epochs):
        tf.logging.info('############################## starting epoch {} #########################'.format(i+1))
        lm.train(input_fn=lambda: input_fn('train', 4, hp.input_maxlen, hp.label_maxlen, pny_lst, han_lst, input_vocab,
                                           label_vocab),
                 hooks=None,
                 steps=None,
                 max_steps=None,
                 saving_listeners=None)
        lm.evaluate(input_fn=lambda: input_fn('eval', 4, hp.input_maxlen, hp.label_maxlen, dev_pny_lst, dev_han_lst, input_vocab,
                                           label_vocab),
                    steps=None,  # Number of steps for which to evaluate model.
                    hooks=None,
                    checkpoint_path=None,  # latest checkpoint in model_dir is used.
                    name=None)


def main(argv):
    train()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main, argv=[sys.argv[0]])