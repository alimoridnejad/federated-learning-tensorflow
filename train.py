from definitions import OUTPUT_DIR, LOG_DIR, TF_RECORDS_DIR, CONFIG_PATH, DATA_DIR
from data import generate_query_batch, make_tfrecords, make_dataset_from_tfrecords
from model import RankNet
from utils import load_yaml_config_file, print_config, get_logger, get_summary_writer
import tensorflow as tf

config = load_yaml_config_file(CONFIG_PATH)
print_config(config)
logger = get_logger(log_dir=LOG_DIR, log_name=config["log_name"])


def train(algo, learning_rate, optimizer, epochs, sigma, dropout_rate, grad_clip, clip_value,
          window_size, batch_size, graph_mode):
    """
    Train and validate RankNet learning to rank model using MSLR dataset.
    :param algo:
    :param learning_rate:
    :param optimizer:
    :param epochs:
    :param sigma:
    :param dropout_rate:
    :param grad_clip:
    :param clip_value:
    :param window_size:
    :param batch_size:
    :param graph_mode:
    :return:
    """

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

    # get the summary writer for train and test
    train_writer, test_writer = get_summary_writer(summary_path=str(LOG_DIR))

    # initialize the model
    model = RankNet(algo=algo,
                    optimizer_type=optimizer,
                    activation=tf.nn.relu,
                    learning_rate=learning_rate,
                    sigma=sigma,
                    dropout_rate=dropout_rate,
                    grad_clip=grad_clip,
                    clip_value=clip_value,
                    ckpt_path=str(OUTPUT_DIR),
                    train_writer=train_writer,
                    test_writer=test_writer,
                    logger=logger,
                    )

    # train the model on train data and validate it using test set
    model.fit(dataset=[train_dataset, test_dataset], epochs=epochs, graph_mode=graph_mode)


if __name__ == "__main__":
    train(
        algo=config["training_algo"],
        optimizer=config["optimizer"],
        learning_rate=config["learning_rate"],
        epochs=config["epochs"],
        sigma=config["sigma"],
        dropout_rate=config["dropout_rate"],
        grad_clip=config["grad_clip"],
        clip_value=config["clip_value"],
        window_size=config["window_size"],
        batch_size=config["batch_size"],
        graph_mode=config["graph_mode"]
    )
