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

from tensorflow_core.python.keras.callbacks import ReduceLROnPlateau

from models.HAN import HAN
from models.AttentionBiRNN import TextAttBiRNN
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('agg')

now = datetime.now()

suffix = now.strftime("%H%M%d")
parser = argparse.ArgumentParser(
        description='Training options',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--model', required=True, type=str,
                    help='Name of the DNN model to train')
parser.add_argument('--data_path', required=True, type=str,
                    help='Path to load tokenized data')
parser.add_argument('--save_dir', required=True, type=str,
                    help='Enter path to the save model checkpoint')

options = parser.parse_args()

with open(f'{options.data_path}/y_train.pickle', 'rb') as handle:
    Y_train = pickle.load(handle)
with open(f'{options.data_path}/y_test.pickle', 'rb') as handle:
    Y_test = pickle.load(handle)
with open(f'{options.data_path}/y_valid.pickle', 'rb') as handle:
    Y_valid = pickle.load(handle)

with open(f'{options.data_path}/x_train.pickle', 'rb') as handle:
    X_train = pickle.load(handle)
with open(f'{options.data_path}/x_test.pickle', 'rb') as handle:
    X_test = pickle.load(handle)
with open(f'{options.data_path}/x_valid.pickle', 'rb') as handle:
    X_valid = pickle.load(handle)
with open(f'{options.data_path}/vocab_set.pickle', 'rb') as handle:
    vocabulary_set = pickle.load(handle)


X_train = tf.constant(X_train)
Y_train = tf.constant(Y_train)
X_test = tf.constant(X_test)
Y_test = tf.constant(Y_test)
X_valid = tf.constant(X_valid)
Y_valid = tf.constant(Y_valid)

N_TRAIN = 1000000
N_TEST = 49000
N_VALID = 49000

X_train = X_train[:N_TRAIN]
Y_train = Y_train[:N_TRAIN]
X_test = X_test[:N_TEST]
Y_test = Y_test[:N_TEST]
X_valid = X_valid[:N_VALID]
Y_valid = Y_valid[:N_VALID]

print("train shape:", X_train.shape, Y_train.shape)
print("X_test shape:", X_test.shape, Y_test.shape)
print("X_valid shape:", X_valid.shape, Y_valid.shape)

print("-----------Data stats:----------")
print("Train buggy rate: {:.2f} %".format(np.sum(Y_train) * 100 / len(Y_train)))
print("Test buggy rate: {:.2f} %".format(np.sum(Y_test) * 100 / len(Y_test)))
print("Validation buggy rate: {:.2f} %".format(np.sum(Y_valid) * 100 / len(Y_valid)))
# Encode training, valid and test instances
encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)

print(f'Vocab size: {encoder.vocab_size}')

strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

with strategy.scope():
    learning_rate = 1e-4
    # Model Definition
    if options.model == "HAN":
        model = HAN(feature_size=encoder.vocab_size, max_sents=7, max_sent_length=500, embedding_dim=64).get_model()
    elif options.model == "TextAttBiRNN":
        model = TextAttBiRNN(maxlen=1000, feature_size=encoder.vocab_size, embedding_dims=64).get_model()
        learning_rate = 1e-3
    elif options.model == "Regularized":
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(encoder.vocab_size, 64),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(0.001), activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(0.001), activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
    elif options.model == "baseline":
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(encoder.vocab_size, 64),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=True)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

    METRICS = [
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.AUC(name='auc'),
    ]
    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate),
                  metrics=METRICS)

model.summary()
batch_size = 512


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


# train_gen = CustomGenerator(X_train, Y_train, batch_size)
# valid_gen = CustomGenerator(X_valid, Y_valid, batch_size)
# test_gen = CustomGenerator(X_test, Y_test, batch_size)

# Training the model
dir_path = f'output/{options.save_dir}_{suffix}'
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
checkpoint_path = os.path.join(dir_path, 'model-{epoch:02d}-{val_loss:.5f}.hdf5')
checkpointer = ModelCheckpoint(checkpoint_path,
                               monitor='val_auc',
                               verbose=1,
                               save_best_only=True,
                               mode='max')

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=3, min_lr=1e-6)
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_auc',
    verbose=1,
    patience=5,
    mode='max',
    restore_best_weights=True)
callback_list = [checkpointer, early_stopping]  # , , reduce_lr
# his1 = model.fit_generator(
#     generator=train_gen,
#     epochs=5,
#     validation_data=valid_gen,
#     callbacks=callback_list)

history = model.fit(X_train, Y_train,
                    batch_size=batch_size,
                    epochs=20,
                    validation_data=(X_valid, Y_valid),
                    callbacks=callback_list)
# predictions = model.predict_generator(test_gen, verbose=1)
loss, accuracy, test_auc = model.evaluate(X_test, Y_test, verbose=2)
print("Test accuracy: {:5.2f}%, auc: {:5.2f}".format(100 * accuracy, test_auc))

predictions = model.predict(X_test)
# pred_indices = tf.math.argmax(predictions, 1)

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
plt.close()


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title(f'Model {options.model} accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig(f'{dir_path}/{options.model}_accuracy.png')
plt.close()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title(f'Model {options.model} loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig(f'{dir_path}/{options.model}_loss.png')
plt.close()

# summarize history for AUC
plt.plot(history.history['auc'])
plt.plot(history.history['val_auc'])
plt.title(f'Model {options.model} AUC')
plt.ylabel('AUC')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig(f'{dir_path}/{options.model}_auc.png')
plt.close()
