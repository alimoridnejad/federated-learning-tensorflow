from datetime import datetime
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import tensorflow as tf
import yaml
from tensorflow.python.framework import tensor_shape

from metrics import NdcgMetric


def get_logger(log_dir, log_name, log_level=logging.INFO):
    """
    This method configures the logger for a file and the console.
    :param log_dir: 
    :param log_name: 
    :param log_level: 
    :return: 
    """

    # create root logger, set format and level
    logger = logging.getLogger(__name__)
    msg_format = '%(asctime)s %(levelname)-8s %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S%p'
    log_formatter = logging.Formatter(fmt=msg_format, datefmt=date_format)
    logger.setLevel(level=log_level)

    # log message to file
    file_handler = logging.FileHandler(filename=f"{log_dir}/{log_name}.log")
    file_handler.setFormatter(fmt=log_formatter)
    logger.addHandler(hdlr=file_handler)

    # print message to the console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt=log_formatter)
    logger.addHandler(hdlr=console_handler)

    return logger


def load_yaml_config_file(yaml_path: Path) -> Dict:
    """
    This method reads and loads the yaml configuration file.
    :param yaml_path: config file path
    :return:
    """
    with open(yaml_path) as file:
        # parse yaml file and produce python dictionary
        config = yaml.safe_load(file)

    return config


def print_config(config_file: Dict) -> None:
    """
    This method prints all the experiment parameters for training.
    :param config_file:
    :return:
    """
    print("=" * 20 + "\n" + "Experiment Parameters" + "\n" + "=" * 20)
    for param, value in config_file.items():
        print(f"{param}: {value}")


class LayerNormalization(tf.keras.layers.Layer):
    """
    Normalize the activations of the previous layer for each given example
    in a batch independently, rather than across a batch like Batch Normalization.
    i.e. applies a transformation that maintains the mean activation within
    each example close to 0 and the activation standard deviation close to 1.

    Given a tensor inputs, moments are calculated and normalization is performed
    across the axes specified in axis.

                        yi = gamma*(xi - mu)/std + beta

    Arguments
        axis: Integer, the axis that should be normalized.
        epsilon: Small float added to variance to avoid dividing by zero.
        beta: beta weight used to center(offset) normalized tensor.
        gamma: gamma weight used to scale normalized tensor


    Input shape
        Arbitrary.
    Output shape
        Same shape as input.
    References:
        - Layer normalization layer (Ba et al., 2016).
            https://arxiv.org/pdf/1607.06450.pdf
    """

    def __init__(self):
        super(LayerNormalization, self).__init__()
        self.hidden_size = None
        self.gamma = None
        self.beta = None

    def build(self, input_shape):
        """
        This method creates the state of the layer (weights).
        :param input_shape:
        :return:
        """
        self.hidden_size = tensor_shape.dimension_value(input_shape[-1])
        self.gamma = self.add_weight(
            name="layer_norm_scale",
            shape=[self.hidden_size],
            dtype="float32",
            initializer=tf.ones_initializer(),
            experimental_autocast=False,
        )
        self.beta = self.add_weight(
            name="layer_norm_bias",
            shape=[self.hidden_size],
            dtype="float32",
            initializer=tf.zeros_initializer(),
            experimental_autocast=False,
        )
        super(LayerNormalization, self).build(input_shape)

    def call(self, x, epsilon=1e-6, input_dtype=tf.float32):
        """
        This method defines the computation from inputs to outputs.
        :param x: input
        :param epsilon:
        :param input_dtype:
        :return:
        """
        mean = tf.reduce_mean(
            input_tensor=x,
            axis=[-1],
            keepdims=True,
            name="mean"
        )
        variance = tf.reduce_mean(
            input_tensor=tf.math.square(x - mean),
            axis=[-1],
            keepdims=True,
            name="variance"
        )
        normalized_input = (x - mean) * tf.math.rsqrt(variance + epsilon)
        return tf.cast(self.gamma * normalized_input + self.beta, dtype=input_dtype)


def get_ndcg(target: tf.Tensor, pred_score: tf.Tensor) -> (float, float):
    """
    This method computes the ndcg@10 and ndcg@30.
    :param target: original document labels
    :param pred_score: computed relevance scores predicted by the model
    :return:
    """
    # pass [-1] to flatten target tensor into 1-D
    target = tf.reshape(target, [-1])

    # create tuple of (target, prediction score)
    tuples = list(zip(target.numpy(), pred_score.numpy()))

    # sort by the predictions (descending)
    tuples.sort(key=lambda x: x[1], reverse=True)

    # get back the ranking list predicted by the model
    pred_rank, _ = list(zip(*tuples))

    # compute ndcg metric for k = [10, 30]
    ndcg_10 = NdcgMetric(k=10).evaluate(rels=pred_rank)
    ndcg_30 = NdcgMetric(k=30).evaluate(rels=pred_rank)

    return ndcg_10, ndcg_30


def log_model_summary_data(writer, step, loss, ndcg_10, ndcg_30) -> None:
    """
    This methods captures the ranking metrics and loss into summary writer.
    :param writer:
    :param step: int
    :param loss:
    :param ndcg_10:
    :param ndcg_30:
    :return:
    """
    with writer.as_default():
        # TensorBoard tage will be the name prefixed by the name scopes
        with tf.name_scope('RankMetrics'):
            tf.summary.scalar(name="loss", data=loss, step=step)
            tf.summary.scalar(name="ndcg@10", data=ndcg_10, step=step)
            tf.summary.scalar(name="ndcg@30", data=ndcg_30, step=step)


def get_summary_writer(summary_path: str) \
        -> Tuple[tf.summary.SummaryWriter, tf.summary.SummaryWriter]:
    """
    This method creates summary file writer for train and test for the
    given log directory.
    :param summary_path:
    :return:
    """
    # sets up a timestamped log directory.
    timestamped = datetime.now().strftime("%Y%m%d-%H%M%S")
    train_summary_path = summary_path + "/train-" + timestamped
    test_summary_path = summary_path + "/test-" + timestamped

    with tf.name_scope('summary'):
        # create two file writers for the log directory.
        train_writer = tf.summary.create_file_writer(train_summary_path)
        test_writer = tf.summary.create_file_writer(test_summary_path)

        return train_writer, test_writer
