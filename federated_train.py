import tensorflow as tf
import tensorflow_federated as tff

from data import generate_query_batch, make_tfrecords, \
    make_dataset_from_tfrecords
from definitions import LOG_DIR, TF_RECORDS_DIR, DATA_DIR
from model_minimal import model_fn
from utils import get_logger

logger = get_logger(log_dir=LOG_DIR, log_name="ranknet")


def train(window_size, batch_size):

    # generate the tf records from raw data
    if not TF_RECORDS_DIR.is_dir():
        logger.info(f"Converting raw data from {DATA_DIR} to tf record files")
        make_tfrecords(data_dir=DATA_DIR, tf_records_dir=TF_RECORDS_DIR)
        logger.info(f"tf records generated and stored at {TF_RECORDS_DIR}")

    # get the MSLR dataset from tf records
    mslr_data_dict = make_dataset_from_tfrecords(tf_records_dir=TF_RECORDS_DIR)
    train_dataset = mslr_data_dict["train"]
    test_dataset = mslr_data_dict["test"]
    train_dataset = generate_query_batch(train_dataset, window_size, batch_size)
    test_dataset = generate_query_batch(test_dataset, window_size, batch_size)

    # take out query ids from dataset
    train_dataset = train_dataset.map(lambda qid, features, label: (features, label))
    test_dataset = test_dataset.map(lambda qid, features, label: (features, label))

    # create two client dataset
    federated_train_data = [train_dataset.take(250), train_dataset.skip(250)]
    federated_test_data = [test_dataset.take(250), test_dataset.skip(250)]

    # construct a pair of federated computations: `initialize` and `next`
    iterative_process = tff.learning.build_federated_averaging_process(
        model_fn=model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0)
    )

    # visualize the metrics from these federated computations using Tensorboard
    logdir = "/tmp/logs/scalars/training/"
    summary_writer = tf.summary.create_file_writer(logdir)

    # invoke the initialize computation to construct the server state
    state = iterative_process.initialize()

    NUM_ROUNDS = 50
    for round_num in range(1, NUM_ROUNDS):
        state, metrics = iterative_process.next(state, federated_train_data)
        print('round {:2d}, metrics={}'.format(round_num, metrics))
        with summary_writer.as_default():
            with tf.name_scope('RankMetrics'):
                for name, value in metrics['train'].items():
                    tf.summary.scalar(name, value, step=round_num)


if __name__ == '__main__':
    train(window_size=512, batch_size=128)
