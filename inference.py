import argparse
import pickle

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.utils import Sequence
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from models.HAN import HAN
from models.AttentionBiRNN import TextAttBiRNN


def compute_AUC(Y_test, predictions, options):
    fpr, tpr, _ = roc_curve(Y_test, predictions)
    roc_auc = auc(fpr, tpr)
    print('AUC: ', roc_auc)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")

    plt.savefig(f'test_auc_model_{options.model}.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Testing options',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--checkpoint', required=True, type=str,
                        help='Enter path to the model checkpoint')
    parser.add_argument('--model', required=True, type=str,
                        help='Name of the DNN model to run test')


    options = parser.parse_args()

    if options.model == 'HAN':
        data_suffix = "HAN"
    else:
        data_suffix = "lstm"

    with open(f'data/x_test_{data_suffix}.pickle', 'rb') as handle:
        X_test = pickle.load(handle)

    with open(f'data/y_test_{data_suffix}.pickle', 'rb') as handle:
        Y_test = pickle.load(handle)
    with open(f'data/vocab_set_{data_suffix}.pickle', 'rb') as handle:
        vocabulary_set = pickle.load(handle)

    N_TEST = 25000
    X_test = X_test[:N_TEST]
    Y_test = np.array(Y_test[:N_TEST])

    encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)
    if options.model == "HAN":
        model = HAN(feature_size=encoder.vocab_size, max_sents=3, max_sent_length=200, embedding_dim=64).get_model()
    elif options.model == "TextAttBiRNN":
        model = TextAttBiRNN(maxlen=1000, feature_size=encoder.vocab_size, embedding_dims=64).get_model()

    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(1e-4),
                  metrics=['accuracy'])

    model.load_weights(options.checkpoint)

    batch_size = 16
    loss, accuracy = model.evaluate(X_test, Y_test, verbose=2)

    print("Restored model, accuracy: {:5.2f}%".format(100 * accuracy))

    predictions = model.predict(X_test)

    compute_AUC(Y_test, predictions, options)
