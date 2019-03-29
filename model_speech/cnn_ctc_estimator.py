import tensorflow as tf
# from tensorflow.python.keras import backend as K

#tf.estimator.Estimator
class AMEstimator(tf.estimator.Estimator):

    def __init__(self, vocab_size, mode, label_vocabulary, model_dir, params, config=None):
        self.vocab_size = vocab_size
        # self.gpu_nums = args.gpu_nums
        self.mode = mode
        self.label_vocabulary = label_vocabulary
        self._config = config
        self._model_init(model_dir=model_dir, params=params, config=config)
        # self._model_init_keras(model_dir, config)

    def model_fn(self, features, labels, mode, params):
        # input_layer = tf.feature_column.input_layer(features=features, feature_columns=params['feature_columns'])
        from tensorflow.python.keras import Input
        input_layer = Input(name='the_inputs', shape=(None, 200, 1), tensor=features['the_inputs'])
        # input_layer = tf.reshape(features['the_inputs'], shape=(-1, -1, 200, 1))
        cnn_1 = self.cnn_cell(32, input_layer)
        cnn_2 = self.cnn_cell(64, cnn_1)
        cnn_3 = self.cnn_cell(128, cnn_2)
        cnn_4 = self.cnn_cell(128, cnn_3, pool=False)
        cnn_5 = self.cnn_cell(128, cnn_4, pool=False)
        # 200 / 8 * 128 = 3200
        # net = tf.reshape(cnn_5, (-1, 3200))
        # net = tf.expand_dims(net, 1)
        net = tf.keras.layers.Reshape(target_shape=(-1, 3200))(cnn_5)

        net = tf.layers.Dropout(0.2)(net)
        net = self.dense(256)(net)
        net = tf.layers.Dropout(0.2)(net)

        logits = self.dense(self.vocab_size, activation=None)(net)
        input_length = tf.cast(tf.squeeze(features['input_length'], axis=-1), tf.int32)
        # decoded_dense, log_prob = K.ctc_decode(logits, input_length, greedy = True, beam_width=10, top_paths=1)

        # logits_softmax = tf.nn.softmax(logits)
        y_pred = tf.transpose(logits, perm=[1, 0, 2])



        decoded, log_prob = tf.nn.ctc_beam_search_decoder(
            inputs=y_pred,
            sequence_length=input_length,
            beam_width=100,
            top_paths=1,
            merge_repeated=False)
        decoded_dense = tf.sparse.to_dense(decoded[0], default_value=-1)
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                # 'text_ids': tf.sparse_to_dense(decoded[0].indices, decoded[0].dense_shape, decoded[0].values, default_value=-1),
                'text_ids': decoded_dense,
                'probabilities': log_prob,
                'logits': logits,
                "input_length": input_length,
                'y_true': tf.sparse.to_dense(tf.to_int32(features['the_labels']), default_value=-1)
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        # label_length = tf.to_int32(tf.squeeze(features['label_length'], axis=-1))
        # sequence_length = tf.to_int32(tf.squeeze(features['input_length'], axis=-1))
        # sparse_labels = tf.to_int32(K.ctc_label_dense_to_sparse(features['the_labels'], label_length))
        sparse_labels = tf.to_int32(features['the_labels'])

        # y_pred = tf.log(tf.transpose(logits, perm=[1, 0, 2]) + 1e-7)
        loss_out = tf.nn.ctc_loss(
            inputs=logits, labels=sparse_labels, sequence_length=input_length, time_major=False)


        # loss_out = K.ctc_batch_cost(features['the_labels'], logits, features['input_length'], features['label_length'])
        loss = tf.reduce_mean(loss_out)


        # Inaccuracy: label error rate

        ler = tf.metrics.mean(tf.edit_distance(tf.cast(decoded[0], tf.int32),
                                              tf.cast(features['the_labels'], tf.int32)))
        metrics = {'label_error_rate': ler}
        tf.summary.scalar('label_error_rate', ler[1])

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode, loss=loss,
                eval_metric_ops=metrics
            )

        assert mode == tf.estimator.ModeKeys.TRAIN

        # rate = tf.train.exponential_decay(0.0008, tf.train.get_global_step(), 10000, 0.01)
        # rate = tf.train.inverse_time_decay(0.0008, tf.train.get_global_step(), 10000, 0.01, staircase=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999, epsilon=10e-8)
        # optimizer = tf.train.AdamOptimizer(0.001)

        grads_and_vars = optimizer.compute_gradients(loss)
        # grad_check = tf.check_numerics(clipped_gradients)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())

        for grad, var in grads_and_vars:
            tf.summary.histogram(var.name + '/gradient', grad)

        # train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    def _model_init(self, model_dir, params, config):
        tf.estimator.Estimator._assert_members_are_not_overridden = lambda self: None
        super(AMEstimator, self).__init__(
            model_fn=self.model_fn,
            model_dir=model_dir,
            params=params,
            config=config)

    def _model_init_keras(self, model_dir, config):
        # 1.声学模型训练-----------------------------------
        from model_speech.cnn_ctc import Am, am_hparams
        am_args = am_hparams()
        am_args.vocab_size = len(self.label_vocabulary)
        am_args.gpu_nums = 0
        am_args.lr = 0.0008
        am_args.is_training = True
        am = Am(am_args)
        base_model = am.ctc_model
        self.ctc_model = tf.keras.estimator.model_to_estimator(base_model, model_dir=model_dir, config=config)

    def conv2d(self, size):
        return tf.layers.Conv2D(size, (3, 3), use_bias=True, activation='relu',
                                padding='same',
                                # kernel_initializer='he_normal'
                                kernel_initializer=tf.contrib.layers.variance_scaling_initializer()
                                )

    def norm(self, x):
        return tf.layers.BatchNormalization(axis=-1)(x)

    def maxpool(self, x):
        return tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding="valid")(x)

    def dense(self, units, activation="relu"):
        return tf.layers.Dense(units, activation=activation, use_bias=True,
                               # kernel_initializer='he_normal'
                               kernel_initializer=tf.contrib.layers.variance_scaling_initializer()
                                )

    def cnn_cell(self, size, x, pool=True):
        x = self.conv2d(size)(x)
        x = self.conv2d(size)(x)
        # x = self.norm(self.conv2d(size)(x))
        # x = self.norm(self.conv2d(size)(x))
        if pool:
            x = self.maxpool(x)
        return x
