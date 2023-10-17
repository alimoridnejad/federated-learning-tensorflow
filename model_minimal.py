import tensorflow as tf
import tensorflow_federated as tff

from metrics import NdcgMetric

tf.executing_eagerly()


class RankNetMinimal(tf.keras.Model):
    def __init__(self):
        super(RankNetMinimal, self).__init__()
        self.ln1 = tf.keras.layers.LayerNormalization()
        self.dense1 = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.ln2 = tf.keras.layers.LayerNormalization()
        self.dense2 = tf.keras.layers.Dense(units=128, activation=tf.nn.relu)
        self.fc_drop = tf.keras.layers.Dropout(rate=0.2)
        self.ln3 = tf.keras.layers.LayerNormalization()
        self.output_layer = tf.keras.layers.Dense(units=1, activation=tf.identity, use_bias=False)

    def call(self, inputs, training=None, mask=None):
        inputs = tf.cast(inputs, tf.float32)
        output = self.dense1(self.ln1(inputs))
        output = self.dense2(self.ln2(output))
        output = self.fc_drop(self.ln3(output), training=training)
        score = self.output_layer(output)
        return score


class CustomRankNetLoss(tf.keras.losses.Loss):
    def __init__(self, name="custom_ranknet_loss"):
        super(CustomRankNetLoss, self).__init__(name=name)
        self.dtype = tf.float32

    def call(self, y_true, y_pred):

        y_true = tf.cast(y_true, self.dtype)
        y_pred = tf.cast(y_pred, self.dtype)
        sigma = 1.0

        # compute P-_ij = 0.5 (1 + S_ij), or (known probability)
        diff_matrix = y_true - tf.transpose(y_true)
        label_diff_matrix = tf.maximum(tf.minimum(1., diff_matrix), -1.)

        # compute P_ij = 1 / (1 + exp(-sigma * (si - sj))),(learned probability)
        pred_diff_matrix = y_pred - tf.transpose(y_pred)

        # C = 0.5*(1 - S_ij)*sigma*(si - sj) + log(1 + exp(-sigma * (si - sj)))
        loss = 1 / 2 * (1 - label_diff_matrix) * sigma * pred_diff_matrix + \
               tf.math.log(1 + tf.exp(-sigma * pred_diff_matrix))

        return tf.reduce_mean(loss)


class CustomNdcgMetric(tf.keras.metrics.Metric):
    """Encapsulates metric logic and state.

    Args:
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.
      **kwargs: Additional layer keywords arguments.

    Subclasses implementation:
    * `__init__()`: All state variables should be created in this method by
      calling `self.add_weight()` like: `self.var = self.add_weight(...)`
    * `update_state()`: Has all updates to the state variables like:
      self.var.assign_add(...).
    * `result()`: Computes and returns a value for the metric
      from the state variables.

    """
    def __init__(self, name="custom_ndcg_metric", dtype=tf.float32, **kwargs):
        super(CustomNdcgMetric, self).__init__(name=name, dtype=dtype, **kwargs)
        self.ndcg30 = self.add_weight(name="ndcg", initializer=tf.constant_initializer(0.0))

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates statistics for the metric.

        Note: This function is executed as a graph function in graph mode.
        This means:
          a) Operations on the same resource are executed in textual order.
             This should make it easier to do things like add the updated
             value of a variable to another, for example.
          b) You don't need to worry about collecting the update ops to execute.
             All update ops added to the graph by this function will be executed.
          As a result, code should generally work the same way with graph or
          eager execution.

        Args:
          *args:
          **kwargs: A mini-batch of inputs to the Metric.
        """

        import ipdb; ipdb.set_trace()

        # pass [-1] to flatten target tensor into 1-D
        y_true = tf.reshape(y_true, [-1])

        y_true = tf.cast(y_true, self.dtype)
        y_pred = tf.cast(y_pred, self.dtype)

        # sort by the predictions (descending)
        indices = tf.argsort(values=y_pred, axis=-1, direction="DESCENDING")

        # get back the ranking list predicted by the model
        pred_rank = tf.gather(params=y_true, indices=indices, axis=-1)

        # compute ndcg metric for k = 30
        ndcg30 = NdcgMetric(k=30).evaluate(rels=pred_rank)

        ndcg30 = tf.cast(ndcg30, self.dtype)

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            ndcg30 = tf.multiply(ndcg30, sample_weight)

        self.ndcg30.assign_add(ndcg30)

        return self.ndcg30

    def result(self):
        """Computes and returns the metric value tensor.

        Result computation is an idempotent operation that simply calculates the
        metric value using the state variables.
        """
        return self.ndcg30

    def reset_states(self):
        """Resets all of the metric state variables.

        This function is called between epochs/steps,
        when a metric is evaluated during training.
        """
        self.ndcg30.assign(0.0)


def create_keras_model_1():
    initializer = tf.keras.initializers.GlorotNormal(seed=0)
    return tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(136,)),
        tf.keras.layers.Dense(1, activation=tf.identity, kernel_initializer=initializer),
    ])


def create_keras_moedel_2():
    seq = tf.keras.Sequential()
    seq.add(tf.keras.layers.InputLayer(input_shape=(136,), dtype=tf.float32))
    seq.add(tf.keras.layers.LayerNormalization())
    seq.add(tf.keras.layers.Dense(units=256, activation=tf.nn.relu))
    seq.add(tf.keras.layers.LayerNormalization())
    seq.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
    seq.add(tf.keras.layers.LayerNormalization())
    seq.add(tf.keras.layers.Dropout(rate=0.2))
    seq.add(tf.keras.layers.Dense(units=1, activation=tf.identity, use_bias=False))
    return seq


def model_fn():
    keras_model = create_keras_moedel_2()
    return tff.learning.from_keras_model(
        keras_model=keras_model,
        input_spec=(tf.TensorSpec(shape=[None, None], dtype=tf.float32),
                    tf.TensorSpec(shape=[None, None], dtype=tf.float32)),
        loss=CustomRankNetLoss(),
        metrics=[tf.keras.metrics.RootMeanSquaredError()])


if __name__ == '__main__':
    iterative_process = tff.learning.build_federated_averaging_process(
        model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0))

