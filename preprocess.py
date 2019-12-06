import argparse
import math
import os
import pickle
import numpy as np
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAXLEN = 500
N_TRAIN = 1000000
N_TEST = 49000
N_VALID = 49000

parser = argparse.ArgumentParser(
    description='Preprocessing options',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--model', required=True, type=str,
                    help='Name of the DNN model to prepare data for.')
parser.add_argument('--data_path', required=True, type=str,
                    help='Path to load tokenized data')

options = parser.parse_args()

# Loading tokenized data
with open(f'data/tokenized_train_clean.pickle', 'rb') as handle:
    train = pickle.load(handle)
with open(f'data/tokenized_valid_clean.pickle', 'rb') as handle:
    valid = pickle.load(handle)
with open(f'data/tokenized_test_clean.pickle', 'rb') as handle:
    test = pickle.load(handle)

train = train[:N_TRAIN]
test = test[:N_TEST]
valid = valid[:N_VALID]

print(train.shape)
print(test.shape)
print(valid.shape)

# Build vocabulary and encoder from the training instances
vocabulary_set = {'<START>', '<END>', '<EMPTY>'}
for index, row in train.iterrows():
    vocabulary_set.update(row['context_before'])
    vocabulary_set.update(row['instance'])
    vocabulary_set.update(row['context_after'])

# Encode training, valid and test instances
encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)


def generate_chunk(items, num_ctx_lines, append=True):
    if not len(items):
        return [""]*num_ctx_lines

    chunk_list = []
    step = int(math.ceil(len(items) / num_ctx_lines))
    num_items_to_add = int(math.ceil(len(items) / num_ctx_lines) * num_ctx_lines - len(items))
    for i in range(num_items_to_add):
        if append:
            items.append("")
        else:
            items.insert(0, "")
    for i in range(0, len(items), step):
        chunk_list.append(" ".join(items[i: i + step]))
    return chunk_list


def reshape_HAN(df, num_context_lines=1):
    N_SAMPLES = df.shape[0]
    X_HAN = np.zeros((N_SAMPLES, 2 * num_context_lines + 1, MAXLEN))
    Y_HAN = []
    idx = 0

    for index, rows in df.iterrows():
        if len(row.instance) <= 1:
            continue
        encodings = []
        lines = generate_chunk(rows["context_before"], num_context_lines, append=False)
        lines.append(" <START> " + " ".join(rows["instance"]) + " <END> ")
        lines = lines + generate_chunk(rows["context_after"], num_context_lines)

        # encodings.append(encoder.encode(" ".join(rows["context_before"])))
        # encodings.append(encoder.encode(" <START> " + " ".join(rows["instance"]) + " <END> "))
        # encodings.append(encoder.encode(" ".join(rows["context_after"])))
        for line in lines:
            encodings.append(encoder.encode(line))
        encodings = pad_sequences(encodings, maxlen=MAXLEN, padding='pre', truncating='pre')

        X_HAN[idx] = encodings
        Y_HAN.append(rows.is_buggy)
        idx = idx + 1

    return X_HAN, Y_HAN


def reshape_instances_normal(df):
    df["input"] = df["context_before"].apply(lambda x: " ".join(x)) + " <START> " + df["instance"].apply(
        lambda x: " ".join(x)) + " <END> " + df["context_after"].apply(lambda x: " ".join(x))
    X_df = []
    Y_df = []
    for index, rows in df.iterrows():
        X_df.append(rows.input)
        Y_df.append(rows.is_buggy)
    return X_df, Y_df


if options.model == 'HAN':
    num_context_lines = 3
    X_train, Y_train = reshape_HAN(train, num_context_lines)
    X_test, Y_test = reshape_HAN(test, num_context_lines)
    X_valid, Y_valid = reshape_HAN(valid, num_context_lines)
else:
    X_train, Y_train = reshape_instances_normal(train)
    X_test, Y_test = reshape_instances_normal(test)
    X_valid, Y_valid = reshape_instances_normal(valid)

    X_train = list(map(lambda x: encoder.encode(x), X_train))
    X_test = list(map(lambda x: encoder.encode(x), X_test))
    X_valid = list(map(lambda x: encoder.encode(x), X_valid))

    X_train = pad_sequences(X_train, maxlen=MAXLEN * 2)
    X_test = pad_sequences(X_test, maxlen=MAXLEN * 2)
    X_valid = pad_sequences(X_valid, maxlen=MAXLEN * 2)

print("X_train", X_train.shape)

print("Saving data...")
if not os.path.exists(options.data_path):
    os.makedirs(options.data_path)

with open(f'{options.data_path}/y_train.pickle', 'wb') as handle:
    pickle.dump(Y_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(f'{options.data_path}/y_test.pickle', 'wb') as handle:
    pickle.dump(Y_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(f'{options.data_path}/y_valid.pickle', 'wb') as handle:
    pickle.dump(Y_valid, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(f'{options.data_path}/x_train.pickle', 'wb') as handle:
    pickle.dump(X_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(f'{options.data_path}/x_test.pickle', 'wb') as handle:
    pickle.dump(X_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(f'{options.data_path}/x_valid.pickle', 'wb') as handle:
    pickle.dump(X_valid, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(f'{options.data_path}/vocab_set.pickle', 'wb') as handle:
    pickle.dump(vocabulary_set, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Done...")
