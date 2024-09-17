import numpy as np
import tensorflow as tf
from keras.layers import Input, GRU, Dense, Embedding, Concatenate, Dot, Softmax, MultiHeadAttention, Add, \
    LayerNormalization
from keras.models import Model
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the Aspect-Sentiment-Intent Model
class AspectSentimentIntentModel:
    def __init__(self, vocab_size, embedding_dim, num_heads, num_blocks, gru_units, intent_units):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.gru_units = gru_units
        self.intent_units = intent_units
        self.model = self._build_model()

    def _build_model(self):
        input_s_t_minus_1 = Input(shape=(None,), name='input_s_t_minus_1')
        input_s_t = Input(shape=(None,), name='input_s_t')

        embedding_layer = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, mask_zero=True)
        embedded_s_t_minus_1 = embedding_layer(input_s_t_minus_1)
        embedded_s_t = embedding_layer(input_s_t)

        def transformer_block(inputs):
            attn_output = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.embedding_dim)(inputs, inputs)
            attn_output = Add()([inputs, attn_output])
            attn_output = LayerNormalization()(attn_output)
            feed_forward_output = Dense(units=self.embedding_dim, activation='relu')(attn_output)
            feed_forward_output = Dense(units=self.embedding_dim)(feed_forward_output)
            return Add()([attn_output, feed_forward_output])

        encoder_s_t_minus_1 = embedded_s_t_minus_1
        encoder_s_t = embedded_s_t

        for _ in range(self.num_blocks):
            encoder_s_t_minus_1 = transformer_block(encoder_s_t_minus_1)
            encoder_s_t = transformer_block(encoder_s_t)

        def average_pooling(embedded_sequence):
            return tf.reduce_mean(embedded_sequence, axis=1)

        aspect_embedding_s_t_minus_1 = average_pooling(encoder_s_t_minus_1)
        aspect_embedding_s_t = average_pooling(encoder_s_t)

        concatenated_aspect_embeddings = Concatenate()([aspect_embedding_s_t_minus_1, aspect_embedding_s_t])

        intent_layer = Dense(self.intent_units, activation='relu')(concatenated_aspect_embeddings)
        intent_vector = Dense(self.embedding_dim)(intent_layer)

        sentiment_input = Concatenate()([aspect_embedding_s_t, intent_vector])
        sentiment_output = Dense(self.vocab_size, activation='softmax')(sentiment_input)

        model = Model(inputs=[input_s_t_minus_1, input_s_t], outputs=sentiment_output)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        return model

    def train(self, input_s_t_minus_1_data, input_s_t_data, target_data, epochs=10, batch_size=32):
        self.model.fit([input_s_t_minus_1_data, input_s_t_data], target_data, epochs=epochs, batch_size=batch_size)

    def predict(self, new_input_s_t_minus_1, new_input_s_t):
        return self.model.predict([new_input_s_t_minus_1, new_input_s_t])

    def summary(self):
        self.model.summary()