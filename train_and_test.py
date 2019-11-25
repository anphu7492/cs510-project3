import os
import sys
import pickle
import argparse
import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import Sequence
from sklearn.metrics import roc_curve, auc
from datetime import datetime

from models.HAN import HAN
from models.AttentionBiRNN import TextAttBiRNN
import matplotlib.pyplot as plt

now = datetime.now()

suffix = now.strftime("%H%M%d")
parser = argparse.ArgumentParser(
        description='Training options',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--model', required=True, type=str,
                    help='Name of the DNN model to train')

parser.add_argument('--save_dir', required=True, type=str,
                    help='Enter path to the save model checkpoint')

options = parser.parse_args()

data_suffix = "HAN" if options.model == "HAN" else "lstm"

with open(f'data/y_train_{data_suffix}.pickle', 'rb') as handle:
    Y_train = pickle.load(handle)
with open(f'data/y_test_{data_suffix}.pickle', 'rb') as handle:
    Y_test = pickle.load(handle)
with open(f'data/y_valid_{data_suffix}.pickle', 'rb') as handle:
    Y_valid = pickle.load(handle)

with open(f'data/x_train_{data_suffix}.pickle', 'rb') as handle:
    X_train = pickle.load(handle)
with open(f'data/x_test_{data_suffix}.pickle', 'rb') as handle:
    X_test = pickle.load(handle)
with open(f'data/x_valid_{data_suffix}.pickle', 'rb') as handle:
    X_valid = pickle.load(handle)
with open(f'data/vocab_set_{data_suffix}.pickle', 'rb') as handle:
    vocabulary_set = pickle.load(handle)

N_TRAIN = 50000
N_TEST = 25000
N_VALID = 25000

Y_train = np.array(Y_train)
Y_test = np.array(Y_test)
Y_valid = np.array(Y_valid)

X_train = X_train[:N_TRAIN]
Y_train = Y_train[:N_TRAIN]
X_test = X_test[:N_TEST]
Y_test = Y_test[:N_TEST]
X_valid = X_valid[:N_VALID]
Y_valid = Y_valid[:N_VALID]

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("X_valid shape:", X_valid.shape)

print("-----------Data stats:----------")
print("Train buggy rate: {:.2f} %".format(np.sum(Y_test) * 100 / len(Y_test)))
print("Test buggy rate: {:.2f} %".format(np.sum(Y_test) * 100 / len(Y_test)))
print("Validation buggy rate: {:.2f} %".format(np.sum(Y_valid) * 100 / len(Y_test)))
# Encode training, valid and test instances
encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)

print(f'Vocab size: {encoder.vocab_size}')

# Model Definition
if options.model == "HAN":
    model = HAN(feature_size=encoder.vocab_size, max_sents=3, max_sent_length=200, embedding_dim=64).get_model()
elif options.model == "TextAttBiRNN":
    model = TextAttBiRNN(maxlen=1000, feature_size=encoder.vocab_size, embedding_dims=64).get_model()
elif options.model == "baseline":
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(encoder.vocab_size, 64),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

model.summary()
batch_size = 16


# Building generators
class CustomGenerator(Sequence):
    def __init__(self, text, labels, batch_size, num_steps=None):
        self.text, self.labels = text, labels
        self.batch_size = batch_size
        self.len = np.ceil(len(self.text) / float(self.batch_size)).astype(np.int64)
        if num_steps:
            self.len = min(num_steps, self.len)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        batch_x = self.text[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y


train_gen = CustomGenerator(X_train, Y_train, batch_size)
valid_gen = CustomGenerator(X_valid, Y_valid, batch_size)
test_gen = CustomGenerator(X_test, Y_test, batch_size)

# Training the model
dir_path = f'output/{options.save_dir}_{suffix}'
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
checkpoint_path = os.path.join(dir_path, 'model-{epoch:02d}-{val_loss:.5f}.hdf5')
checkpointer = ModelCheckpoint(checkpoint_path,
                               monitor='val_loss',
                               verbose=1,
                               save_best_only=True,
                               mode='min')

callback_list = [checkpointer]  # , , reduce_lr
his1 = model.fit_generator(
    generator=train_gen,
    epochs=1,
    validation_data=valid_gen,
    callbacks=callback_list)

predictions = model.predict_generator(test_gen, verbose=1)
pred_indices = tf.math.argmax(predictions, 1)

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
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")

plt.savefig(f'{dir_path}/auc_model_{options.model}.png')
