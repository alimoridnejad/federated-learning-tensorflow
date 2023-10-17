from model_minimal import CustomNdcgMetric
import tensorflow as tf

mse = tf.keras.metrics.MeanSquaredError()

ndcg30 = CustomNdcgMetric()

ndcg30.update_state(y_true=tf.constant([1., 2., 3.]),
                    y_pred=tf.constant([1., 2., 3.]))

ndcg30.result()

