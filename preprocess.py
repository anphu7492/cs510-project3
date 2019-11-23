import pickle
import pandas as pd
import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAXLEN = 200
N_TRAIN = 100000
N_TEST = 25000
N_VALID = 25000

# Loading tokenized data
with open('data/tokenized_train.pickle', 'rb') as handle:
    train = pickle.load(handle)
with open('data/tokenized_valid.pickle', 'rb') as handle:
   valid = pickle.load(handle)
with open('data/tokenized_test.pickle', 'rb') as handle:
   test = pickle.load(handle)

train = train[:N_TRAIN]
test = test[:N_TEST]
valid = valid[:N_VALID]

print(train.shape)
print(test.shape)
print(valid.shape)

# Build vocabulary and encoder from the training instances
vocabulary_set = {'<START>', '<END>'}
for index, row in train.iterrows():
   vocabulary_set.update(row['context_before'])
   vocabulary_set.update(row['instance'])
   vocabulary_set.update(row['context_after'])

# Encode training, valid and test instances
encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)


def reshape_HAN(df):
    N_SAMPLES = df.shape[0]
    X_HAN = np.zeros((N_SAMPLES, 3, MAXLEN))
    Y_HAN = []
    idx = 0
    for index, rows in df.iterrows():
        encodings = []
        encodings.append(encoder.encode(" ".join(rows["context_before"])))
        encodings.append(encoder.encode(" <START> " + " ".join(rows["instance"]) + " <END> "))
        encodings.append(encoder.encode(" ".join(rows["context_after"])))
        encodings = pad_sequences(encodings, maxlen=MAXLEN, padding='pre', truncating='pre')

        X_HAN[idx] = encodings
        Y_HAN.append(rows.is_buggy)
        idx = idx + 1

    return X_HAN, Y_HAN


X_train, Y_train = reshape_HAN(train)
X_test, Y_test = reshape_HAN(test)
X_valid, Y_valid = reshape_HAN(valid)

print("X_train", X_train.shape)

"""
# Reshape instances:
def reshape_instances(df):
    df["input"] = df["context_before"].apply(lambda x: " ".join(x)) + " <START> " + df["instance"].apply(lambda x: " ".join(x)) + " <END> " + df["context_after"].apply(lambda x: " ".join(x))
    X_df = []
    Y_df = []
    for index, rows in df.iterrows():
        X_df.append(rows.input)
        Y_df.append(rows.is_buggy)
    return X_df, Y_df

X_train, Y_train = reshape_instances(train)
X_test, Y_test = reshape_instances(test)
X_valid, Y_valid = reshape_instances(valid)

# Use a subset of data to save time
# You can change it in Part(III) to improve your result
X_train = X_train[:100000]
Y_train = Y_train[:100000]
X_test = X_test[:25000]
Y_test = Y_test[:25000]
X_valid = X_valid[:25000]
Y_valid = Y_valid[:25000]

# Build vocabulary and encoder from the training instances
vocabulary_set = set()
for data in X_train:
   vocabulary_set.update(data.split())

# Encode training, valid and test instances
encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)

def encode(text):
  encoded_text = encoder.encode(text)
  return encoded_text

X_train = list(map(lambda x: encode(x), X_train))
X_test = list(map(lambda x: encode(x), X_test))
X_valid = list(map(lambda x: encode(x), X_valid))

X_train = pad_sequences(X_train, maxlen=MAXLEN)
X_test = pad_sequences(X_test, maxlen=MAXLEN)
X_valid = pad_sequences(X_valid, maxlen=MAXLEN)
"""

model_name = "HAN"
with open(f'data/y_train_{model_name}.pickle', 'wb') as handle:
    pickle.dump(Y_train, handle)
with open(f'data/y_test_{model_name}.pickle', 'wb') as handle:
    pickle.dump(Y_test, handle)
with open(f'data/y_valid_{model_name}.pickle', 'wb') as handle:
    pickle.dump(Y_valid, handle)
with open(f'data/x_train_{model_name}.pickle', 'wb') as handle:
    pickle.dump(X_train, handle)
with open(f'data/x_test_{model_name}.pickle', 'wb') as handle:
    pickle.dump(X_test, handle)
with open(f'data/x_valid_{model_name}.pickle', 'wb') as handle:
    pickle.dump(X_valid, handle)
with open(f'data/vocab_set_{model_name}.pickle', 'wb') as handle:
    pickle.dump(vocabulary_set, handle)

