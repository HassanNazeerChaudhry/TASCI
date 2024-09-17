import numpy as np
import tensorflow as tf
from keras.layers import Input, GRU, Dense, Embedding, Concatenate, Dot, Softmax, MultiHeadAttention, Add, LayerNormalization
from keras.models import Model
import torch
import torch.nn as nn
import torch.nn.functional as F

class AspectExtractionModel(nn.Module):
    def __init__(self, embed_size, heads, num_classes):
        super(AspectExtractionModel, self).__init__()
        self.self_attention = SelfAttention(embed_size, heads)
        self.fc = nn.Linear(embed_size, num_classes)

    def forward(self, x, mask):
        x = self.self_attention(x, x, x, mask)
        x = x.mean(dim=1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)