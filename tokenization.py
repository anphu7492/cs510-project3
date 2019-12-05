import pandas as pd
import pickle
import re
import javalang
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib; matplotlib.use('agg')

# Load the data:
with open('data/train.pickle', 'rb') as handle:
    train = pickle.load(handle)
with open('data/valid.pickle', 'rb') as handle:
    valid = pickle.load(handle)
with open('data/test.pickle', 'rb') as handle:
    test = pickle.load(handle)

# TODO> remove
# train = train[:1000]
# test = test[:500]
# valid = valid[:500]

# drop duplicate
print('# samples before dropping duplicated (train, test, val):', len(train.index), len(test.index), len(valid.index))
train = train.drop_duplicates(subset=["instance", "context_before", "context_after"])
valid = valid.drop_duplicates(subset=["instance", "context_before", "context_after"])
test = test.drop_duplicates(subset=["instance", "context_before", "context_after"])
print('# samples after dropping duplicated (train, test, val):', len(train.index), len(test.index), len(valid.index))


def remove_comments(source):
    # remove all occurrences streamed comments (/*COMMENT */) from string
    source = re.sub(re.compile("/\*.*?\*/", re.DOTALL), "", source)
    # remove all occurrence single-line comments (//COMMENT\n ) from string
    source = re.sub(re.compile("//.*?\n"), "", source)
    return source


def remove_comments_df(df):
    df['instance'] = df['instance'].apply(lambda x: remove_comments(x))
    df['context_before'] = df['context_before'].apply(lambda x: remove_comments(x))
    df['context_after'] = df['context_after'].apply(lambda x: remove_comments(x))
    return df


train = remove_comments_df(train)
test = remove_comments_df(test)
valid = remove_comments_df(valid)


# Tokenize and shape our input:
def custom_tokenize(string):
    try:
        tokens = list(javalang.tokenizer.tokenize(string))
    except:
        return []
    values = []
    for token in tokens:
        # Abstract strings
        if '"' in token.value or "'" in token.value:
            values.append('$STRING$')
        # Abstract numbers (except 0 and 1)
        elif token.value.isdigit() and int(token.value) > 1:
            values.append('$NUMBER$')
        # other wise: get the value
        else:
            values.append(token.value)
    return values


def tokenize_df(df):
    df['instance'] = df['instance'].apply(lambda x: custom_tokenize(x))
    df['context_before'] = df['context_before'].apply(lambda x: custom_tokenize(x))
    df['context_after'] = df['context_after'].apply(lambda x: custom_tokenize(x))
    return df


test = tokenize_df(test)
train = tokenize_df(train)
valid = tokenize_df(valid)


def compute_length_df(df):
    df['instance_length'] = df['instance'].apply(lambda x: len(x))
    df['context_before_length'] = df['context_before'].apply(lambda x: len(x))
    df['context_after_length'] = df['context_after'].apply(lambda x: len(x))
    return df


train = compute_length_df(train)
test = compute_length_df(test)
valid = compute_length_df(valid)

print(train["context_before_length"].describe())
print(train["instance_length"].describe())
print(train["context_after_length"].describe())

print("==========================Remove non-useful data===============================")
train = train.drop(train[train["context_before"].map(len) < 5].index)
train = train.drop(train[train["instance"].map(len) < 2].index)
train = train.drop(train[train["context_after"].map(len) < 5].index)

print("Num train after remove 0 context:", len(train))
print(train["context_before_length"].describe())
print(train["instance_length"].describe())
print(train["context_after_length"].describe())

fig = plt.figure(figsize=(12, 8))
sns.distplot(train['instance_length'], norm_hist=True)
plt.xlabel("Distribution of instance's length", fontsize=13)
plt.savefig('instance-length-dist_noisy.png')
plt.close()

fig = plt.figure(figsize=(12, 8))
sns.distplot(train['context_before_length'], norm_hist=True)
plt.xlabel("Distribution of context_before_length", fontsize=13)
plt.savefig('context-before-length-dist_noisy.png')
plt.close()

fig = plt.figure(figsize=(12, 8))
sns.distplot(train['context_after_length'], norm_hist=True)
plt.xlabel("Distribution of context_after_length", fontsize=13)
plt.savefig('context-after-length-dist_noisy.png')
plt.close()

print("=========================Remove based on z-score==================")

# https://people.richland.edu/james/lecture/m170/ch08-int.html
train = train[(np.abs(stats.zscore(train[["context_before_length", "instance_length", "context_after_length"]])) < 2.3).all(axis=1)]
# train = train[(np.abs(stats.zscore(train["instance_length"])) < 3)]
# train = train[(np.abs(stats.zscore(train["context_after_length"])) < 3)]

print(train["context_before_length"].describe())
print(train["instance_length"].describe())
print(train["context_after_length"].describe())

print("Context before: mean length={:.3f}, std={:.3f}".format(np.mean(train["context_before_length"]),
                                                              np.std(train["context_before_length"])))
print("Instance: mean length={:.3f}, std={:.3f}".format(np.mean(train["instance_length"]),
                                                        np.std(train["instance_length"])))
print("Context after: mean length={:.3f}, std={:.3f}".format(np.mean(train["context_after_length"]),
                                                             np.std(train["context_after_length"])))

fig = plt.figure(figsize=(12, 8))
sns.distplot(train['instance_length'])
plt.xlabel("Distribution of instance's length", fontsize=13)
plt.savefig('instance-length-dist.png')
plt.close()

fig = plt.figure(figsize=(12, 8))
sns.distplot(train['context_before_length'])
plt.xlabel("Distribution of context_before_length", fontsize=13)
plt.savefig('context-before-length-dist.png')
plt.close()

fig = plt.figure(figsize=(12, 8))
sns.distplot(train['context_after_length'])
plt.xlabel("Distribution of context_after_length", fontsize=13)
plt.savefig('context-after-length-dist.png')
plt.close()

train.drop(columns=["context_before_length", "instance_length", "context_after_length"])
with open('data/tokenized_train_clean.pickle', 'wb') as handle:
    pickle.dump(train, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('data/tokenized_valid_clean.pickle', 'wb') as handle:
    pickle.dump(valid, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('data/tokenized_test_clean.pickle', 'wb') as handle:
    pickle.dump(test, handle, protocol=pickle.HIGHEST_PROTOCOL)
