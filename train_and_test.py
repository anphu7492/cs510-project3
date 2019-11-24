import os
import sys
import pickle
import pandas as pd
import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import Sequence
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.utils import to_categorical
from datetime import datetime

from models.HAN import HAN
from models.AttentionBiRNN import TextAttBiRNN
import matplotlib.pyplot as plt

now = datetime.now()

prefix = sys.argv[1] if len(sys.argv) > 1 else now.strftime("%m%d%y_%H")
model_name = "HAN"
data_suffix = "HAN"
print("prefix:", prefix)

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
N_VALID = 250000

X_train = X_train[:N_TRAIN]
Y_train = Y_train[:N_TRAIN]
X_test = X_test[:N_TEST]
Y_test = Y_test[:N_TEST]
X_valid = X_valid[:N_VALID]
Y_valid = Y_valid[:N_VALID]
# Y_valid = to_categorical(Y_valid)

print("X_train shape:", X_train.shape)
print("Y_train shape:", len(Y_train))
print("Y_test length:", len(Y_test))
# Encode training, valid and test instances
encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)

print(f'Vocab size: {encoder.vocab_size}')

# Model Definition
if model_name == "HAN":
    model = HAN(encoder.vocab_size, max_sents=3, max_sent_length=200, embedding_dim=64).get_model()
elif model_name == "TextAttBiRNN":
    model = TextAttBiRNN(maxlen=1000, max_features=encoder.vocab_size, embedding_dims=64).get_model()
elif model_name == "baseline":
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
save_dir = f'data/models/{model_name}/{prefix}'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
checkpoint_path = os.path.join(save_dir, 'model-{epoch:02d}-{val_loss:.5f}.hdf5')
checkpointer = ModelCheckpoint(checkpoint_path,
                               monitor='val_loss',
                               verbose=1,
                               save_best_only=True,
                               mode='min')

callback_list = [checkpointer]  # , , reduce_lr
his1 = model.fit_generator(
    generator=train_gen,
    epochs=5,
    validation_data=valid_gen,
    callbacks=callback_list)

predictions = model.predict_generator(test_gen, verbose=1)
pred_indices = tf.math.argmax(predictions, 1)

# fpr, tpr, _ = roc_curve(Y_test_orig, predictions[:, 1])
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

plt.savefig(f'{prefix}_auc_model_{model_name}.png')
