from paddle import ParamAttr
from paddle.nn.initializer import KaimingNormal
import numpy as np
import paddle
import paddle.nn as nn
from paddle.nn.initializer import TruncatedNormal, Constant, Normal
import paddle.nn.functional as F
from paddle import Tensor
from einops import rearrange
from typing import Tuple
import math

trunc_normal_ = TruncatedNormal(std=.02)
normal_ = Normal
zeros_ = Constant(value=0.)
ones_ = Constant(value=1.)

class Mlp(nn.Layer):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        _, N, C = x.shape
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def drop_path(x, drop_prob=0., training=False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ...
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = paddle.to_tensor(1 - drop_prob, dtype=x.dtype)
    shape = (paddle.shape(x)[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = keep_prob + paddle.rand(shape, dtype=x.dtype)
    random_tensor = paddle.floor(random_tensor)  # binarize
    output = x.divide(keep_prob) * random_tensor
    return output


class DropPath(nn.Layer):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Identity(nn.Layer):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


def encoder_layer(in_c, out_c, k=3, s=2, p=1):
    return  nn.Sequential(
            nn.Conv2D(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=k,
            stride=s,
            padding=p,
            groups=8,
            weight_attr=paddle.ParamAttr(initializer=nn.initializer.KaimingUniform())),
            nn.ReLU())


def decoder_layer(in_c, out_c, k=3, s=1, p=1, mode='nearest', scale_factor=False, size=False):
    align_corners = False if mode=='nearest' else True
    return nn.Sequential(nn.Upsample(size=size, mode=mode, align_corners=align_corners),
                         nn.Conv2D(
                             in_channels=in_c,
                             out_channels=out_c,
                             kernel_size=k,
                             stride=s,
                             padding=p,
                             groups=8,
                             weight_attr=paddle.ParamAttr(initializer=nn.initializer.KaimingUniform())),
                         nn.ReLU())


class PositionalEncoding(nn.Layer):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = paddle.zeros(shape=[max_len, d_model])
        position = paddle.arange(0, max_len, dtype=np.float32).unsqueeze(1)
        div_term = paddle.exp(paddle.arange(0, d_model, 2).astype(np.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = paddle.sin(position * div_term)
        pe[:, 1::2] = paddle.cos(position * div_term)
        pe = pe.unsqueeze(0)
        pe = paddle.transpose(pe, perm=[1, 0, 2])
        self.register_buffer('pe', pe)

    def forward(self, x):
        x0, x1, x2 = x.shape
        x = x + self.pe[:x0, :]
        return self.dropout(x)

class PositionAttention(nn.Layer):
    def __init__(self, max_length, in_channels=512, num_channels=64,
                 h=8, w=25, mode='nearest', **kwargs):
        super().__init__()
        self.max_length = max_length
        self.k_encoder = nn.Sequential(
            encoder_layer(in_channels, num_channels, s=(1, 2)),
            encoder_layer(num_channels, num_channels, s=(2, 2)),
            encoder_layer(num_channels, num_channels, s=(2, 2)),
            encoder_layer(num_channels, num_channels, s=(2, 2)))
        self.en_ln = nn.Sequential(
            nn.LayerNorm(num_channels),
            nn.LayerNorm(num_channels),
            nn.LayerNorm(num_channels),
            nn.LayerNorm(num_channels))
        hs = [h, h, math.ceil(h / 2), math.ceil(h / 4)]
        ws = [w, math.ceil(w / 2), math.ceil(w / 4), math.ceil(w / 8)]
        self.h = h
        self.w = w
        self.k_decoder = nn.Sequential(
            decoder_layer(num_channels, num_channels, size=(hs[3], ws[3]), mode=mode),
            decoder_layer(num_channels, num_channels, size=(hs[2], ws[2]), mode=mode),
            decoder_layer(num_channels, num_channels, size=(hs[1], ws[1]), mode=mode),
            decoder_layer(num_channels, in_channels, size=(hs[0], ws[0]), mode=mode)
        )
        self.de_ln = nn.Sequential(
            nn.LayerNorm(num_channels),
            nn.LayerNorm(num_channels),
            nn.LayerNorm(num_channels),
            nn.LayerNorm(in_channels))
        self.pos_encoder = PositionalEncoding(in_channels, dropout=0, max_len=max_length)
        self.project = nn.Linear(in_channels, in_channels)

    def forward(self, x, last_vec=None):
        N, L, C = x.shape
        x = x.reshape([N, C, self.h, self.w])
        N, E, H, W = x.shape
        k, v = x, x  # (N, E, H, W)

        features = []
        for i in range(0, len(self.k_encoder)):
            k = self.k_encoder[i](k)
            b, c, h, w = k.shape
            k = k.flatten(2).transpose((0, 2, 1))
            k = self.en_ln[i](k)
            k = k.transpose([0, 2, 1]).reshape([b, c, h, w])
            features.append(k)
        for i in range(0, len(self.k_decoder) - 1):
            k = self.k_decoder[i](k)
            b, c, h, w = k.shape
            k = k.flatten(2).transpose((0, 2, 1))
            k = self.de_ln[i](k)
            k = k.transpose([0, 2, 1]).reshape([b, c, h, w])
            k = k + features[len(self.k_decoder) - 2 - i]
        k = self.k_decoder[-1](k)
        b, c, h, w = k.shape
        k = k.flatten(2).transpose((0, 2, 1))
        k = self.de_ln[-1](k)
        k = k.transpose([0, 2, 1]).reshape([b, c, h, w])


        # TODO q=f(q,k)
        if last_vec is None:
            zeros = paddle.zeros(shape=[self.max_length, N, E], dtype='float32')
        else:
            zeros = paddle.transpose(last_vec, perm=[1, 0, 2]) # (T, N, E)
        q = self.pos_encoder(zeros)  # (T, N, E)
        q = paddle.transpose(q, perm=[1, 0, 2])  # (N, T, E)
        q = self.project(q)  # (N, T, E)
        # calculate attention
        attn_scores = (q.matmul(k.flatten(2, 3)))
        attn_scores = attn_scores / (E ** 0.5)
        attn_scores = nn.functional.softmax(attn_scores, axis=-1)

        v = paddle.transpose(v, perm=[0, 2, 3, 1])
        v = paddle.reshape(v, shape=[N, -1, E]) # (N, (H*W), E)
        attn_vecs = (attn_scores.matmul(v)) # (N, T, E)

        return attn_vecs, attn_scores
