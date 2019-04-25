import os
import tensorflow as tf
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard

from utils import get_data, data_hparams
# from keras.callbacks import ModelCheckpoint


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
#
# # 0.准备验证所需数据------------------------------
# data_args = data_hparams()
# data_args.data_type = 'dev'
# data_args.data_path = 'data/'
# data_args.thchs30 = True
# data_args.aishell = False
# data_args.prime = False
# data_args.stcmd = False
# data_args.batch_size = 4
# data_args.data_length = None
# # data_args.data_length = 10
# data_args.shuffle = True
# dev_data = get_data(data_args)
#
# # 1.声学模型训练-----------------------------------
# from model_speech.cnn_ctc import Am, am_hparams
# am_args = am_hparams()
# am_args.vocab_size = len(train_data.am_vocab)
# am_args.gpu_nums = 1
# am_args.lr = 0.0008
# am_args.is_training = True
# am = Am(am_args)
#
# if os.path.exists('logs_am_new/model.h5'):
#     print('load acoustic model...')
#     am.ctc_model.load_weights('logs_am_new/model.h5')
#
# epochs = 1
batch_num = len(train_data.wav_lst) // train_data.batch_size
#
# # checkpoint
# ckpt = "model_{epoch:02d}-{val_acc:.2f}.hdf5"
# checkpoint = ModelCheckpoint(os.path.join('./checkpoint', ckpt), monitor='val_loss', save_weights_only=False, verbose=1, save_best_only=True)
#
# tbCallBack = TensorBoard(log_dir='./logs',  # log 目录
#                          histogram_freq=1,  # 按照何等频率（epoch）来计算直方图，0为不计算
#                          batch_size=4,     # 用多大量的数据计算直方图
#                          write_graph=True,  # 是否存储网络结构图
#                          write_grads=True, # 是否可视化梯度直方图
#                          write_images=True,# 是否可视化参数
#                          embeddings_freq=0,
#                          embeddings_layer_names=None,
#                          embeddings_metadata=None)
# #
# # for k in range(epochs):
# #     print('this is the', k+1, 'th epochs trainning !!!')
# #     batch = train_data.get_am_batch()
# #     dev_batch = dev_data.get_am_batch()
# #     am.ctc_model.fit_generator(batch, steps_per_epoch=batch_num, epochs=10, callbacks=[checkpoint], workers=1, use_multiprocessing=False, validation_data=dev_batch, validation_steps=200)
#
# batch = train_data.get_am_batch()
# dev_batch = dev_data.get_am_batch()
#
# am.ctc_model.fit_generator(batch, steps_per_epoch=batch_num, epochs=epochs, callbacks=[checkpoint, tbCallBack], workers=1, use_multiprocessing=False, validation_data=dev_batch, validation_steps=200)
# am.ctc_model.save_weights('logs_am_new/model.h5')


# 2.语言模型训练-------------------------------------------
from model_language.transformer import Lm, lm_hparams
lm_args = lm_hparams()
lm_args.num_heads = 8
lm_args.num_blocks = 6
lm_args.input_vocab_size = len(train_data.pny_vocab)
lm_args.label_vocab_size = len(train_data.han_vocab)
lm_args.max_length = 100
lm_args.hidden_units = 512
lm_args.dropout_rate = 0.2
lm_args.lr = 0.0003
lm_args.is_training = True
lm = Lm(lm_args)

epochs = 10
with lm.graph.as_default():
    saver =tf.train.Saver()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(graph=lm.graph, config=config) as sess:
    merged = tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())
    add_num = 0
    if os.path.exists('logs_lm/checkpoint'):
        print('loading language model...')
        latest = tf.train.latest_checkpoint('logs_lm')
        add_num = int(latest.split('_')[-1])
        saver.restore(sess, latest)
    writer = tf.summary.FileWriter('logs_lm/tensorboard', tf.get_default_graph())
    for k in range(epochs):
        total_loss = 0
        batch = train_data.get_lm_batch()
        for i in range(batch_num):
            input_batch, label_batch = next(batch)
            feed = {lm.x: input_batch, lm.y: label_batch}
            cost,_ = sess.run([lm.mean_loss,lm.train_op], feed_dict=feed)
            total_loss += cost
            if (k * batch_num + i) % 10 == 0:
                rs=sess.run(merged, feed_dict=feed)
                writer.add_summary(rs, k * batch_num + i)
        print('epochs', k+1, ': average loss = ', total_loss/batch_num)
    saver.save(sess, 'logs_lm/model_%d' % (epochs + add_num))
    writer.close()
