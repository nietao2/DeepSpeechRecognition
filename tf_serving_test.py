import grpc
import tensorflow as tf
import numpy as np

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import base64

from model_speech.cnn_ctc_dataset import load_vocab, load_data, AudioDataset

label_vocab = load_vocab(['./data/thchs_train.txt', './data/thchs_dev.txt', './data/thchs_test.txt'])
wav_lst, pny_lst = load_data(['./data/thchs_train.txt'], './data/', size=2)

dataset = AudioDataset(wav_lst, pny_lst, label_vocab, 2)



channel = grpc.insecure_channel('localhost:8500')
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
request = predict_pb2.PredictRequest()
request.model_spec.name = 'am'
request.model_spec.signature_name = 'serving_default'


examples = []
for inputs, label in dataset._generator_fn():
    example = tf.train.Example(
        features = tf.train.Features(
            feature = {
                'the_inputs':tf.train.Feature(float_list = tf.train.FloatList(value=np.array(inputs['the_inputs'].flatten(), dtype=float))),
                'input_length':tf.train.Feature(int64_list = tf.train.Int64List(value = [inputs['input_length']])),
            }))
    serialized = example.SerializeToString()
    examples.append(serialized)

request.inputs['examples'].CopyFrom(
    tf.contrib.util.make_tensor_proto(examples))
result = stub.Predict(request, 5.0)  # 5 seconds
print(result.outputs['text_ids'].int64_val)
text = []
for i in result.outputs['text_ids'].int64_val:
    text.append(label_vocab[i])
text = ' '.join(text)
print('文本结果：', text)




