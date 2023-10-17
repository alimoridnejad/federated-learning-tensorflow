import logging
import time
from collections import defaultdict
from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense

from definitions import OUTPUT_DIR, LOG_DIR
from utils import LayerNormalization, log_model_summary_data, get_ndcg, get_logger

# In Tensorflow 2.0, eager execution is enabled by default.
# Debug in eager mode, then decorate with @tf.function
tf.executing_eagerly()

# The @tf.function trace-compiles train_step into a TF graph for faster
# execution. The function specializes to the precise shape of the argument
# tensors. To avoid re-tracing due to the variable sequence lengths or variable
# batch sizes (the last batch is smaller), use input_signature to specify
# more generic shapes.
# use tf.TensorSpec to add metadata for describing the `tf.Tensor` objects
train_step_signature = [
    # mslr dataset has 136 features. output is a single doc
    tf.TensorSpec(shape=(None, 136), dtype=tf.float32),
    tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
]


class RankNet(tf.keras.Model):
    """
    This class defines a feedforward neural network with appropriate layers as
    per given network structure.

    Args:
        algo (str): normal or factorized version of RankNet model
        optimizer_type (str): optimization type
        activation (tf.nn.relu): activation function
        learning_rate (float):
        sigma (float): parameter σ determines the shape of the sigmoid.
        dropout_rate (float): Fraction of the input units to drop. Float between 0 and 1.
        grad_clip (bool): whether to clip gradients or not
        clip_value (float): used to clip gard tensor values to this specified min and max.
        ndcg (float): Normalized Discounted Cumulative Gain
        ckpt_path (str): path to model checkpoints
        train_writer (tf.summary.SummaryWriter): train summary file writer
        test_writer (tf.summary.SummaryWriter): test summary file writer
        logger (logging): logger

    Attributes:
        algo
        activation
        optimizer_type (tf.keras.optimizers)
        learning_rate
        sigma
        dropout_rate
        grad_clip
        clip_value
        train_fuc (tf.function): train function
        test_fuc (tf.function): test function
        ndcg
        ckpt_path
        ckpt_manager (tf.train.CheckpointManager): Manages multiple checkpoints
        train_writer
        test_writer

    """
    def __init__(self,
                 algo: str = "default",
                 optimizer_type: str = "adam",
                 activation: tf.nn.relu = tf.nn.relu,
                 learning_rate: float = 0.001,
                 sigma: float = 1.0,
                 dropout_rate: float = 0.25,
                 grad_clip: bool = True,
                 clip_value: float = 1.0,
                 ndcg: float = 0.0,
                 ckpt_path: str = str(OUTPUT_DIR),
                 train_writer: tf.summary.SummaryWriter = None,
                 test_writer: tf.summary.SummaryWriter = None,
                 logger: logging = get_logger(log_dir=LOG_DIR, log_name="ranknet"),
                 ):
        super(RankNet, self).__init__()

        self.algo = algo
        self.learning_rate = learning_rate
        self.optimizer = self._create_optimizer(optimizer_type=optimizer_type)
        self.activation = activation
        self.sigma = sigma
        self.dropout_rate = dropout_rate
        self.grad_clip = grad_clip
        self.clip_value = clip_value
        self.train_fuc = None
        self.test_fuc = None
        self.ndcg = ndcg
        self.ckpt_path = ckpt_path
        self.logger = logger
        self.ckpt_manager = self._create_checkpoint_manager(checkpoint_path=self.ckpt_path)
        self.train_writer = train_writer
        self.test_writer = test_writer

        # TODO: add a loop to iterate on any arbitrary units for net structure

        hidden_units = [256, 128]

        # apply layer normalization
        self.ln1 = LayerNormalization()
        self.dense1 = Dense(256, activation=self.activation)

        self.ln2 = LayerNormalization()
        self.dense2 = Dense(128, activation=self.activation)

        # add a dropout layer
        self.fc_drop = tf.keras.layers.Dropout(self.dropout_rate)

        self.ln3 = LayerNormalization()
        self.output_layer = Dense(1, activation=tf.identity, use_bias=False)

    def call(self, inputs: Any, training: bool = False) -> tf.Tensor:
        """
        This method implements the model's forward pass.
        :param inputs:
        :param training: use to specify a different behavior in training and inference
        :return:
        """
        inputs = tf.cast(inputs, tf.float32)

        output = self.dense1(self.ln1(inputs))
        output = self.dense2(self.ln2(output))

        output = self.fc_drop(self.ln3(output), training=training)
        score = self.output_layer(output)
        return score

    def _create_optimizer(self, optimizer_type: str) -> tf.keras.optimizers:
        """
        This method sets the training optimizer based on the selected type.
        :param optimizer_type:
        :return:
        """

        with tf.name_scope("optimizer"):
            if optimizer_type == "adam":
                self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
            else:
                self.optimizer = tf.keras.optimizers.SGD(self.learning_rate)
            return self.optimizer

    def _create_checkpoint_manager(self, checkpoint_path: str, max_to_keep: int = 5,
                                   load_model: bool = False) -> tf.train.CheckpointManager:
        """
        This method manages multiple checkpoints by keeping some and deleting unneeded ones.
        :param checkpoint_path:
        :param max_to_keep:
        :param load_model:
        :return:
        """
        with tf.name_scope('checkpoint_manager'):
            checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self)
            self.ckpt_manager = tf.train.CheckpointManager(checkpoint=checkpoint,
                                                           directory=checkpoint_path,
                                                           max_to_keep=max_to_keep)
            if load_model:
                checkpoint.restore(self.ckpt_manager.latest_checkpoint)
                self.logger.info("Latest model checkpoint restored!!")
            else:
                self.logger.info("Start instantiating and building the model ...")

            return self.ckpt_manager

    @staticmethod
    def _get_lambda_scaled_derivative(tape, pred_score, Wk, lambdas: tf.Tensor) -> tf.Tensor:
        """
        This method computes gradient used to update the weights wk.

                                ∂C/∂wk = λij * (∂si/∂wk - ∂sj/∂wk)
        where λij
                        λij = 1/2 * (1−Sij) − 1 / (1 + exp(sigma * (si−sj)))

        :param tape:
        :param pred_score: targets
        :param Wk: model weights
        :param lambdas: factorization λij
        :return:

        Ref: Burges, Christopher JC. "From ranknet to lambdarank to lambdamart:
        An overview." Learning 11, no. 23-581 (2010): 81.
        """

        # ∂si/∂wk
        dsi_dWk = tape.jacobian(pred_score, Wk)

        # ∂si/∂wk−∂sj/∂wk
        dsi_dWk_minus_dsj_dWk = tf.expand_dims(dsi_dWk, 1) - tf.expand_dims(dsi_dWk, 0)

        shape = tf.concat([tf.shape(lambdas),
                           tf.ones([tf.rank(dsi_dWk_minus_dsj_dWk) - tf.rank(lambdas)],
                                   dtype=tf.int32)], axis=0)

        # ∂C/∂wk = λij (∂si/∂wk - ∂sj/∂wk)
        grad = tf.reshape(lambdas, shape) * dsi_dWk_minus_dsj_dWk
        grad = tf.reduce_mean(grad, axis=[0, 1])

        return grad

    def _get_ranknet_loss(self, pred_score: tf.Tensor, real_score: tf.Tensor) -> tf.Tensor:
        """
        This method computes cross entropy loss function for pairwise ranking.
        :param pred_score:
        :param real_score:
        :return:
        Ref: Burges, Christopher JC. "From ranknet to lambdarank to lambdamart:
        An overview." Learning 11, no. 23-581 (2010): 81.
        """
        
        with tf.name_scope("ranknet_loss"):

            # compute P-_ij = 0.5 (1 + S_ij), or (known probability)
            diff_matrix = real_score - tf.transpose(real_score)
            label_diff_matrix = tf.maximum(tf.minimum(1., diff_matrix), -1.)

            # compute P_ij = 1 / (1 + exp(-sigma * (si - sj))), or (learned probability)
            pred_diff_matrix = pred_score - tf.transpose(pred_score)

            # C = 0.5 * (1 - S_ij) * sigma * (si - sj) + log(1 + exp(-sigma * (si - sj)))
            loss = 1 / 2 * (1 - label_diff_matrix) * self.sigma * pred_diff_matrix + \
                   tf.math.log(1 + tf.exp(-self.sigma * pred_diff_matrix))

        return loss

    def _get_lambdas(self, pred_score: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
        """
        This method computes lambdas based on the following equation:

                    λij = 1/2 * (1−Sij) − 1 / (1 + exp(sigma * (si−sj)))

        The model computes the scores si = f(xi) and sj = f(xj)
        parameter sigma determines the shape of the sigmoid
        For a given query:
        S_ij ∈ {0,±1}: 1 if relevant, -1 if irrelevant, 0 if scores are equal

        :param pred_score:
        :param labels:
        :return:

        Ref: Eq. 3 in Burges, Christopher JC. "From ranknet to lambdarank to lambdamart:
        An overview." Learning 11, no. 23-581 (2010): 81.
        """

        with tf.name_scope("lambdas"):

            # build Sij matrix based on actual labels
            diff_matrix = labels - tf.transpose(labels)
            label_diff_matrix = tf.maximum(tf.minimum(1., diff_matrix), -1.)

            # build (si - sj) based on model predictions
            pred_diff_matrix = pred_score - tf.transpose(pred_score)

            # lambdas = 1/2 * (1−Sij) − 1 / (1 + exp(sigma * (si−sj)))
            lambdas = self.sigma * ((1 / 2) * (1 - label_diff_matrix) -
                                    tf.nn.sigmoid(-self.sigma * pred_diff_matrix))

        return lambdas

    def _train_step(self, inputs: tf.Tensor, target: tf.Tensor):
        """
        This method makes one step train for a non-factorized training algorithm.
        :param inputs:
        :param target:
        :return:
        """
        # record operations for automatic differentiation
        with tf.GradientTape() as tape:
            pred_score = self(inputs, training=tf.constant(True))
            loss = tf.reduce_mean(self._get_ranknet_loss(pred_score=pred_score, real_score=target))
            pred_score = tf.reshape(pred_score, [-1])

        # compute the gradient using operations recorded in context of this tape
        with tf.name_scope("gradients"):
            gradients = tape.gradient(target=loss, sources=self.trainable_variables)
            if self.grad_clip:
                gradients = [(tf.clip_by_value(t=grad,
                                               clip_value_min=-self.clip_value,
                                               clip_value_max=self.clip_value))
                             for grad in gradients]

        # apply gradients to variables
        self.optimizer.apply_gradients(grads_and_vars=zip(gradients, self.trainable_variables))

        return loss, pred_score

    def _factorized_train_step(self, inputs: tf.Tensor, target: tf.Tensor):
        """
        This method makes one step train for a factorized training algorithm.
        :param inputs:
        :param target:
        :return:
        """
        # record operations for automatic differentiation
        with tf.GradientTape(persistent=True) as tape:

            # ensure that tensor is being traced by this tape
            tape.watch(inputs)
            pred_score = self(inputs, training=tf.constant(True))
            loss = tf.reduce_mean(self._get_ranknet_loss(pred_score=pred_score, real_score=target))
            lambdas = self._get_lambdas(pred_score, target)
            pred_score = tf.reshape(pred_score, [-1])

        # compute the gradient using operations recorded in context of this tape
        with tf.name_scope("gradients"):
            gradients = [self._get_lambda_scaled_derivative(tape, pred_score, Wk, lambdas) \
                         for Wk in self.trainable_variables]

            if self.grad_clip:
                gradients = [(tf.clip_by_value(t=grad,
                                               clip_value_min=-self.clip_value,
                                               clip_value_max=self.clip_value))
                             for grad in gradients]

            # apply gradients to variables
            self.optimizer.apply_gradients(grads_and_vars=zip(gradients, self.trainable_variables))

        return loss, pred_score

    @tf.function(input_signature=train_step_signature)
    def train_step(self, inputs, target):
        return self._train_step(inputs, target)

    @tf.function(input_signature=train_step_signature)
    def factorized_train_step(self, inputs, target):
        return self._factorized_train_step(inputs, target)

    def _test_step(self, inputs, target):
        pred_score = self(inputs, training=tf.constant(False))
        loss = tf.reduce_mean(self._get_ranknet_loss(pred_score=pred_score, real_score=target))
        pred_score = tf.reshape(pred_score, [-1])
        return loss, pred_score

    @tf.function(input_signature=train_step_signature)
    def test_step(self, inputs, target):
        return self._test_step(inputs, target)

    def _save_model(self, ndcg):
        if ndcg > self.ndcg:
            ckpt_save_path = self.ckpt_manager.save()
            self.ndcg = ndcg
            self.logger.info(f'Saving checkpoint at {ckpt_save_path}')

    def _init_comp_graph(self, step=0, name=""):
        with self.train_writer.as_default():
            tf.summary.trace_export(
                name=name,
                step=step,
                profiler_outdir=self.log_dir)

    def set_train_test_function(self, graph_mode: bool) -> None:
        """
        This method sets both train and test function based on the graph_mode.
        :param graph_mode:
        :return:
        """
        if graph_mode:
            self.logger.info("Running Model in graph mode.............")
            self.test_fuc = self.test_step
            if self.algo == "default":
                self.train_fuc = self.train_step
            else:
                self.train_fuc = self.factorized_train_step
        else:
            self.logger.info("Running Model in eager mode.............")
            self.test_fuc = self._test_step
            if self.algo == "default":
                self.train_fuc = self._train_step
            else:
                self.train_fuc = self._factorized_train_step

    def fit(self, dataset, epochs, graph_mode=False):
        self.set_train_test_function(graph_mode)

        assert len(dataset) == 2
        train_dataset, test_dataset = dataset
        if graph_mode:
            tf.summary.trace_on(graph=True, profiler=True)

        for epoch in range(epochs):
            print(f"\nStart of epoch {epoch}")
            start_time = time.time()

            train_session_ndcgs = defaultdict(list)
            train_losses = []
            # iterate over the batches of the dataset.
            for train_step, (q_id, inputs, target) in enumerate(train_dataset):
                train_loss, score = self.train_fuc(inputs, target)
                train_losses.append(train_loss)
                ndcg10, ndcg30 = get_ndcg(target, score)
                train_session_ndcgs[10].append(ndcg10)
                train_session_ndcgs[30].append(ndcg30)

                if train_step == 1:
                    if graph_mode:
                        self._init_comp_graph()

                # print loss every 100 batches
                if train_step % 500 == 0:
                    print(f"Training loss (for one batch) at step {train_step}: {train_loss:.3f}")
                    print(f"Seen so far: {(train_step + 1) * 128} samples")

            train_ndcg = {k: np.mean(train_session_ndcgs[k]) for k in [10, 30]}
            train_epoch_loss = np.mean(train_losses)
            log_model_summary_data(self.train_writer,
                                   epoch,
                                   train_epoch_loss,
                                   train_ndcg[10],
                                   train_ndcg[30])

            # display train loss at the end of each epoch.
            print(f"Train loss over epoch: {train_epoch_loss:.3f}")

            test_session_ndcgs = defaultdict(list)
            test_losses = []
            # run a validation loop at the end of each epoch
            for test_step, (qid, inputs_test, target_test) in enumerate(test_dataset):
                test_loss, pred_score = self.test_fuc(inputs_test, target_test)
                test_losses.append(test_loss)
                ndcg10, ndcg30 = get_ndcg(target_test, pred_score)
                test_session_ndcgs[10].append(ndcg10)
                test_session_ndcgs[30].append(ndcg30)

            test_epoch_loss = np.mean(test_losses)
            test_ndcg = {k: np.mean(train_session_ndcgs[k]) for k in [10, 30]}
            log_model_summary_data(self.test_writer,
                                   epoch,
                                   test_epoch_loss,
                                   test_ndcg[10],
                                   test_ndcg[30])

            # display metrics at the end of each epoch
            print(f"Test metrics: NDCG@10: {test_ndcg[10]: .3f}, NDCG@30{test_ndcg[30]: .3f}")
            print(f"Time taken: {(time.time() - start_time): .2f}s")

            # save the model
            self._save_model(test_ndcg[30])
