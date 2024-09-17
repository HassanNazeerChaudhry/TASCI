import numpy as np
import tensorflow as tf
from keras.layers import Input, GRU, Dense, Embedding, Concatenate, Dot, Softmax, MultiHeadAttention, Add, LayerNormalization
from keras.models import Model
import torch
import torch.nn as nn
import torch.nn.functional as F


# Define the GRU-based Sequence-to-Sequence Model
class GRUSeq2SeqModel:
    def __init__(self, vocab_size, embedding_dim, gru_units):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.gru_units = gru_units
        self.model = self._build_model()

    def _build_model(self):
        input_s_t_minus_1 = Input(shape=(None,), name='input_s_t_minus_1')
        input_s_t = Input(shape=(None,), name='input_s_t')

        embedding_layer = tf.keras.layers.Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim,
                                                    mask_zero=True)
        embedded_s_t_minus_1 = embedding_layer(input_s_t_minus_1)
        embedded_s_t = embedding_layer(input_s_t)

        gru_encoder_t_minus_1 = GRU(self.gru_units, return_sequences=True, return_state=True)
        encoder_output_t_minus_1, encoder_state_t_minus_1 = gru_encoder_t_minus_1(embedded_s_t_minus_1)

        gru_encoder_t = GRU(self.gru_units, return_sequences=True, return_state=True)
        encoder_output_t, encoder_state_t = gru_encoder_t(embedded_s_t)

        attention_scores = Dot(axes=[2, 2])([encoder_output_t, encoder_output_t])
        attention_weights = Softmax()(attention_scores)
        context_vector = Dot(axes=[2, 1])([attention_weights, encoder_output_t])

        concat_vector = Concatenate(axis=-1)([context_vector, encoder_state_t])

        decoder_gru = GRU(self.gru_units, return_sequences=False)
        decoder_output = decoder_gru(concat_vector)

        dense_output = Dense(self.vocab_size, activation='softmax')(decoder_output)

        model = Model(inputs=[input_s_t_minus_1, input_s_t], outputs=dense_output)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        return model

    def train(self, input_s_t_minus_1_data, input_s_t_data, target_data, epochs=10, batch_size=32):
        self.model.fit([input_s_t_minus_1_data, input_s_t_data], target_data, epochs=epochs, batch_size=batch_size)

    def predict(self, new_input_s_t_minus_1, new_input_s_t):
        return self.model.predict([new_input_s_t_minus_1, new_input_s_t])

    def summary(self):
        self.model.summary()