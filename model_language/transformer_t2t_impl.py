# coding=utf-8
# Imports we need.
import json

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import collections

from tensor2tensor import models
from tensor2tensor import problems
from tensor2tensor.data_generators import problem, text_problems, translate, generator_utils, text_encoder, cleaner_en_xx
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import t2t_model
from tensor2tensor.utils import registry
from tensor2tensor.utils import metrics
from tensor2tensor.utils import data_reader
from tensor2tensor.utils.trainer_lib import create_run_config, create_experiment

LM_TRAIN_DATASETS = ['../data/thchs_train.txt']
LM_DEV_DATASETS = ['../data/thchs_dev.txt']
LM_TEST_DATASETS = ['../data/thchs_test.txt']

@registry.register_problem('pinyin2zh_problem')
class MyProblem(translate.TranslateProblem):
    @property
    def approx_vocab_size(self):
        return 2**15  # 32k

    @property
    def source_vocab_name(self):
        return "%s.pinying" % self.vocab_filename

    @property
    def target_vocab_name(self):
        return "%s.zh" % self.vocab_filename

    @property
    def source_filenames(self):
        return ['thchs_train.txt', 'thchs_dev.txt', 'thchs_test.txt']

    @property
    def dataset_splits(self):
        """Splits of data to produce and number of output shards for each."""
        return [{
            "split": problem.DatasetSplit.TRAIN,
            "shards": 1,
        }, {
            "split": problem.DatasetSplit.EVAL,
            "shards": 1,
        }, {
            "split": problem.DatasetSplit.TEST,
            "shards": 1,
        }]

    def get_training_dataset(self, tmp_dir):
        """UN Parallel Corpus and CWMT Corpus need to be downloaded manually.

        Append to training dataset if available

        Args:
          tmp_dir: path to temporary dir with the data in it.

        Returns:
          paths
        """
        dataset = LM_TRAIN_DATASETS
        for i in range(len(dataset)):
            dataset[i] = os.path.join(tmp_dir, dataset[i])
        return dataset


    def generate(self, tmp_dir, source_filenames, index):
        for source_filename in source_filenames:
            filepath = os.path.join(tmp_dir, source_filename)
            tf.logging.info("Generating vocab from %s", filepath)
            with tf.gfile.GFile(filepath, mode="r") as source_file:
                for line in source_file:
                    line = line.strip()
                    if line and "\t" in line:
                        parts = line.split("\t")
                        part = parts[index].strip()
                        yield part

    def text2text_txt_tab_iterator(self, txt_path):
        """Yield dicts for Text2TextProblem.generate_samples from lines of txt_path.

        Args:
          txt_path: path to txt file with a record per line, source and target
            are tab-separated.

        Yields:
          {"inputs": inputs, "targets": targets}
        """
        for line in text_problems.txt_line_iterator(txt_path):
            if line and "\t" in line:
                parts = line.split("\t")
                inputs, targets = parts[1:3]
                yield {"inputs": inputs.strip(), "targets": targets.strip()}

    def compile_data(self, tmp_dir, datasets, filename, datatypes_to_clean=None):
        """Concatenates all `datasets` and saves to `filename`."""
        datatypes_to_clean = datatypes_to_clean or []
        filename = os.path.join(tmp_dir, filename)
        lang1_fname = filename + ".lang1"
        lang2_fname = filename + ".lang2"
        if tf.gfile.Exists(lang1_fname) and tf.gfile.Exists(lang2_fname):
            tf.logging.info("Skipping compile data, found files:\n%s\n%s", lang1_fname,
                            lang2_fname)
            return filename
        with tf.gfile.GFile(lang1_fname, mode="w") as lang1_resfile:
            with tf.gfile.GFile(lang2_fname, mode="w") as lang2_resfile:
                for lang_filename in datasets:
                    lang_filepath = os.path.join(tmp_dir, lang_filename)
                    is_sgm = (
                            lang_filename.endswith("sgm"))

                    for example in self.text2text_txt_tab_iterator(lang_filepath):
                        line1res = translate._preprocess_sgm(example["inputs"], is_sgm)
                        line2res = translate._preprocess_sgm(example["targets"], is_sgm)
                        clean_pairs = [(line1res, line2res)]
                        if "txt" in datatypes_to_clean:
                            clean_pairs = cleaner_en_xx.clean_en_xx_pairs(clean_pairs)
                        for line1res, line2res in clean_pairs:
                            if line1res and line2res:
                                lang1_resfile.write(line1res)
                                lang1_resfile.write("\n")
                                lang2_resfile.write(line2res)
                                lang2_resfile.write("\n")

        return filename

    def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
        # train_dataset = self.get_training_dataset(tmp_dir)
        if dataset_split == problem.DatasetSplit.TRAIN:
            datasets = LM_TRAIN_DATASETS
            tag = "train"
        elif dataset_split == problem.DatasetSplit.EVAL:
            datasets = LM_DEV_DATASETS
            tag = "dev"
        else:
            datasets = LM_TEST_DATASETS
            tag = "test"

        # train = dataset_split == problem.DatasetSplit.TRAIN

        # datasets = train_dataset if train else LM_TEST_DATASETS
        # source_datasets = [[item[0], [item[1][0]]] for item in train_dataset]
        # target_datasets = [[item[0], [item[1][1]]] for item in train_dataset]
        source_vocab = generator_utils.get_or_generate_vocab_inner(
            data_dir=data_dir,
            vocab_filename=self.source_vocab_name,
            vocab_size=self.approx_vocab_size,
            generator=self.generate(tmp_dir=tmp_dir, source_filenames=self.source_filenames, index=1),
            max_subtoken_length=None

        )
        target_vocab = generator_utils.get_or_generate_vocab_inner(
            data_dir=data_dir,
            vocab_filename=self.target_vocab_name,
            vocab_size=self.approx_vocab_size,
            generator=self.generate(tmp_dir=tmp_dir, source_filenames=self.source_filenames, index=2),
            max_subtoken_length=1
        )
        # tag = "train" if train else "dev"
        filename_base = "thchs_pinyinzh_%sk_tok_%s" % (self.approx_vocab_size, tag)
        data_path = self.compile_data(tmp_dir, datasets, filename_base)
        return text_problems.text2text_generate_encoded(
                    text_problems.text2text_txt_iterator(data_path + ".lang1",
                                                         data_path + ".lang2"),
                    source_vocab, target_vocab)

    def feature_encoders(self, data_dir):
        source_vocab_filename = os.path.join(data_dir, self.source_vocab_name)
        target_vocab_filename = os.path.join(data_dir, self.target_vocab_name)
        source_token = text_encoder.SubwordTextEncoder(source_vocab_filename)
        target_token = text_encoder.SubwordTextEncoder(target_vocab_filename)
        return {
            "inputs": source_token,
            "targets": target_token,
        }


def main():
    # print(registry.list_hparams())
    data_dir = '../t2t_data/'
    tmp_dir = '../data/'
    TRAIN_DIR = '../logs_lm_new_t2t'
    MODEL = 'transformer'
    PROBLEM = 'pinyin2zh_problem'

    tfe = tf.contrib.eager
    tfe.enable_eager_execution()

    pinyin2zh_problem = registry.problem(PROBLEM)
    pinyin2zh_problem.generate_data(data_dir=data_dir, tmp_dir=tmp_dir)
    hparams = trainer_lib.create_hparams("transformer_base")
    hparams.batch_size = 4
    hparams.learning_rate_warmup_steps = 45000
    hparams.learning_rate = 0.0003
    print(json.loads(hparams.to_json()))




    # Initi Run COnfig for Model Training
    RUN_CONFIG = create_run_config(
        model_name=MODEL,
        model_dir=TRAIN_DIR # Location of where model file is store
        # More Params here in this fucntion for controling how noften to tave checkpoints and more.
    )

    # Create Tensorflow Experiment Object
    tensorflow_exp_fn = create_experiment(
        run_config=RUN_CONFIG,
        hparams=hparams,
        model_name=MODEL,
        problem_name=PROBLEM,
        data_dir=data_dir,
        train_steps=400000, # Total number of train steps for all Epochs
        eval_steps=100 # Number of steps to perform for each evaluation
    )

    # Kick off Training
    tensorflow_exp_fn.train_and_evaluate()

if __name__ == '__main__':
    main()