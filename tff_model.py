import collections
from typing import NamedTuple, OrderedDict

import tensorflow as tf
import tensorflow_federated as tff

from model_minimal import RankNetMinimal


def create_federated_ranknet_variables() -> NamedTuple:
    """
    This method defines the tensorflow model variables such as weights and bias
    that we will train, as well as variables that will hold various cumulative statistics
    such as num_examples and loss_sum. It creates data structure to represent the entire set.
    :return:
    """
    ranknet_variables = collections.namedtuple(typename="ranknet_variables",
                                               field_names="weights bias num_examples loss_sum")

    return ranknet_variables(
        weights=tf.Variable(initial_value=lambda: tf.ones(shape=(136,), dtype=tf.float32),
                            name="weights",
                            trainable=True),
        bias=tf.Variable(initial_value=lambda: tf.zeros(shape=(136,), dtype=tf.float32),
                         name="bias",
                         trainable=True),
        num_examples=tf.Variable(initial_value=0.0, name='num_examples', trainable=False),
        loss_sum=tf.Variable(initial_value=0.0, name='loss_sum', trainable=False),
    )


def federated_ranknet_forward_pass(variables, batch):
    """
    This method defines the forward pass method that computes loss, emits predictions,
    and updates the cumulative statistics for a single batch of input data
    :param variables:
    :param batch:
    :return:
    """

    q_id, inputs, target = batch["q_id"], batch["x"], batch["y"]
    model = RankNetMinimal()
    # import ipdb; ipdb.set_trace()
    loss , pred_score = model.train_step(inputs, target)

    num_examples = tf.cast(tf.size(batch["y"]), tf.float32)

    variables.num_examples.assign_add(num_examples)
    variables.loss_sum.assign_add(loss * num_examples)

    return loss, pred_score


def get_local_ranknet_metrics(variables) -> OrderedDict:
    """
    This method returns a set of local metrics that are eligible to be aggregated to the server
    in a federated learning or evaluation process.
    We'll need num examples to correctly weight the contributions from different users
    when computing federated aggregates.
    :param variables:
    :return:
    """
    return collections.OrderedDict(
      num_examples=variables.num_examples,
      loss=variables.loss_sum / variables.num_examples)


@tff.federated_computation
def aggregate_ranknet_metrics_across_clients(metrics) -> OrderedDict:
    """
    This method defines how to aggregate the local metrics emitted by each device via get_local_mnist_metrics
    :param metrics: the OrderedDict returned by get_local_ranknet_metrics
    :return: dictionary of global aggregates defines the set of metrics which will be available on the server.
    """
    return collections.OrderedDict(
      num_examples=tff.federated_sum(metrics.num_examples),
      loss=tff.federated_mean(metrics.loss, metrics.num_examples))


class FedRanknetModel(tff.learning.Model):

    def __init__(self):
        self._variables = create_federated_ranknet_variables()

    @property
    def trainable_variables(self):
        """
        variables that can and should be trained using gradient-based methods
        :return:
        """
        return [self._variables.weights, self._variables.bias]

    @property
    def non_trainable_variables(self):
        """
        variables that can be fixed pre-trained layers, or static model data
        :return:
        """
        return []

    @property
    def local_variables(self):
        return [
            self._variables.num_examples, self._variables.loss_sum,
        ]

    @property
    def input_spec(self):
        """
        A nested structure of `tf.TensorSpec` objects, that matches the structure of
        arguments that will be passed as the `batch_input` argument of
        `forward_pass`. The tensors must include a batch dimension as the first
        dimension, but the batch dimension may be undefined.
        :return:
        """
        return collections.OrderedDict(
            q_id=tf.TensorSpec([None, 1], tf.float32),
            x=tf.TensorSpec([None, 136], tf.float32),
            y=tf.TensorSpec([None, 1], tf.float32))

    @tf.function
    def forward_pass(self, batch, training=True):
        del training
        loss, predictions = federated_ranknet_forward_pass(self._variables, batch)
        num_exmaples = tf.shape(batch["x"])[0]
        return tff.learning.BatchOutput(
            loss=loss, predictions=predictions, num_examples=num_exmaples)

    @tf.function
    def report_local_outputs(self):
        return get_local_ranknet_metrics(self._variables)

    @property
    def federated_output_computation(self):
        return aggregate_ranknet_metrics_across_clients

