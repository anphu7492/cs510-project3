import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import initializers, layers
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import Embedding, Bidirectional, GRU, TimeDistributed
from tensorflow.keras.models import Model


class AttLayer(layers.Layer):
    def __init__(self, attention_dim):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = tf.Variable(self.init((input_shape[-1], self.attention_dim)), name='W')
        self.b = tf.Variable(self.init((self.attention_dim, )), name='b')
        self.u = tf.Variable(self.init((self.attention_dim, 1)), name='u')
        self.trainable_att_weights = [self.W, self.b, self.u]
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'attention_dim': self.attention_dim,
        })
        return config


class HAN(object):
    def __init__(self, vocab_size, max_sent_length=200, max_sents=15, embedding_dim=100):
        self.vocab_size = vocab_size
        self.max_sent_length = max_sent_length
        self.max_sentences = max_sents
        self.embedding_dim = embedding_dim

    def get_model(self):
        embedding_layer = Embedding(self.vocab_size,
                                    self.embedding_dim)
                                    # weights=[embedding_matrix],
                                    # input_length=self.max_sent_length,
                                    # trainable=True,

        sentence_input = Input(shape=(self.max_sent_length,), dtype='int64')
        embedded_sequences = embedding_layer(sentence_input)
        l_lstm = Bidirectional(GRU(100, return_sequences=True))(embedded_sequences)
        l_att = AttLayer(100)(l_lstm)
        # l_att = l_lstm
        sentEncoder = Model(sentence_input, l_att)

        print("Sentence input", sentence_input.shape)
        print("Embedded sequences", embedded_sequences.shape)
        print("l_lstm", l_lstm.shape)
        print("l_att", l_att.shape)

        review_input = Input(shape=(self.max_sentences, self.max_sent_length), dtype='int64')
        print("review_input", review_input.shape)
        review_encoder = TimeDistributed(sentEncoder)(review_input)
        print("review_encoder", review_encoder.shape)
        gru = GRU(100, return_sequences=True)
        l_lstm_sent = Bidirectional(gru)(review_encoder)
        # l_lstm_sent = Bidirectional(GRU(300, return_sequences=True))(review_encoder)
        print("l_lstm_sent", l_lstm_sent.shape)
        l_att_sent = AttLayer(100)(l_lstm_sent)
        print("l_att_sent", l_att_sent.shape)
        preds = Dense(2, activation='softmax')(l_att_sent)
        model = Model(review_input, preds)

        return model
