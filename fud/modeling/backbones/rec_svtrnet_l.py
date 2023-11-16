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

from rec_utils import PositionAttention

trunc_normal_ = TruncatedNormal(std=.02)
normal_ = Normal
zeros_ = Constant(value=0.)
ones_ = Constant(value=1.)
class AdapterLayer(nn.Layer):
    def __init__(self, input_size):
        super(AdapterLayer, self).__init__()
        self.adapter = nn.Sequential(
            nn.Linear(input_size, input_size // 4),
            nn.Hardswish(),
            nn.Linear(input_size // 4, input_size),
        )

    def forward(self, input_features):
        batch_size, seq_length, input_size = input_features.shape
        input_features_reshaped = input_features.reshape([batch_size * seq_length, input_size])
        adapter_output = self.adapter(input_features_reshaped)
        adapter_output = adapter_output.reshape([batch_size, seq_length, input_size])
        return input_features + adapter_output


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


class ConvBNLayer(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 bias_attr=False,
                 groups=1,
                 act=nn.GELU):
        super().__init__()
        self.conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            weight_attr=paddle.ParamAttr(initializer=nn.initializer.KaimingUniform()),
            bias_attr=bias_attr)
        self.norm = nn.BatchNorm2D(out_channels)
        self.act = act()

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.norm(out)
        out = self.act(out)
        return out

class ConvLNLayer(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 bias_attr=False,
                 groups=1,
                 act=nn.GELU):
        super().__init__()
        self.conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            weight_attr=paddle.ParamAttr(initializer=nn.initializer.KaimingUniform()),
            bias_attr=bias_attr)
        self.norm = nn.LayerNorm(out_channels)
        self.act = act()

    def forward(self, inputs):
        out = self.conv(inputs)
        b, c, h, w= out.shape
        out = out.flatten(2).transpose((0, 2, 1))
        out = self.norm(out)
        out = out.transpose([0, 2, 1]).reshape([b, c, h, w])
        out = self.act(out)
        return out


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


class Attention(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads=8,
                 mixer='Global',
                 HW=None,
                 local_k=[7, 11],
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 H=8,
                 W=25):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.H = H
        self.W = W
        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.HW = HW
        self.H = H
        self.W = W
        if HW is not None:
            H = HW[0]
            W = HW[1]
            self.N = H * W
            self.C = dim
        if mixer == 'Local' and HW is not None:
            hk = local_k[0]
            wk = local_k[1]
            mask = paddle.ones([H * W, H + hk - 1, W + wk - 1], dtype='float32')
            for h in range(0, H):
                for w in range(0, W):
                    mask[h * W + w, h:h + hk, w:w + wk] = 0.
            mask_paddle = mask[:, hk // 2:H + hk // 2, wk // 2:W + wk //
                               2].flatten(1)
            mask_inf = paddle.full([H * W, H * W], '-inf', dtype='float32')
            mask = paddle.where(mask_paddle < 1, mask_paddle, mask_inf)
            self.mask = mask.unsqueeze([0, 1])
        self.mixer = mixer
        self.lpe = nn.Conv2D(
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim,
            weight_attr=paddle.ParamAttr(initializer=nn.initializer.KaimingUniform()))

    def forward(self, x):
        if self.HW is not None:
            N = self.N
            C = self.C
        else:
            _, N, C = x.shape

        qkv = self.qkv(x).reshape((0, N, 3, self.num_heads, C //
                                   self.num_heads)).transpose((2, 0, 3, 1, 4))
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]

        attn = (q.matmul(k.transpose((0, 1, 3, 2))))
        if self.mixer == 'Local':
            attn += self.mask
        attn = nn.functional.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        if self.mixer == 'Global':
            lpe = self.lpe(rearrange(x, 'n (h w) c -> n c h w', h=self.H, w=self.W))
            lpe = rearrange(lpe, 'n c h w -> n (h w) c')
            x = (attn.matmul(v)).transpose((0, 2, 1, 3)).reshape((0, N, C))
            x = x + lpe

        else:
            x = (attn.matmul(v)).transpose((0, 2, 1, 3)).reshape((0, N, C))

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads,
                 mixer='Global',
                 local_mixer=[7, 11],
                 HW=None,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer='nn.LayerNorm',
                 epsilon=1e-6,
                 prenorm=True,
                 H = 8,
                 W = 25):
        super().__init__()
        if isinstance(norm_layer, str):
            self.norm1 = eval(norm_layer)(dim, epsilon=epsilon)
        else:
            self.norm1 = norm_layer(dim)
        if mixer == 'Global' or mixer == 'Local':
            self.mixer = Attention(
                dim,
                num_heads=num_heads,
                mixer=mixer,
                HW=HW,
                local_k=local_mixer,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                H=H,
                W=W)
        else:
            raise TypeError("The mixer must be one of [Global, Local, Conv]")

        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        if isinstance(norm_layer, str):
            self.norm2 = eval(norm_layer)(dim, epsilon=epsilon)
        else:
            self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_ratio = mlp_ratio
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)
        self.prenorm = prenorm
        self.adaptlayer = AdapterLayer(dim)
        self.dim = dim
        self.h = H
        self.w = W
        self.act1 = nn.Silu()
        self.act2 = nn.Silu()
        if isinstance(norm_layer, str):
            self.ln1 = eval(norm_layer)(dim, epsilon=epsilon)
        else:
            self.ln1 = norm_layer(dim)
        if isinstance(norm_layer, str):
            self.ln2 = eval(norm_layer)(dim, epsilon=epsilon)
        else:
            self.ln2 = norm_layer(dim)
        self.conv1 = nn.Conv2D(
            dim,
            dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups= dim,
            weight_attr=paddle.ParamAttr(initializer=nn.initializer.KaimingUniform()))
        self.conv2 = nn.Conv2D(
            dim,
            dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups= dim,
            weight_attr=paddle.ParamAttr(initializer=nn.initializer.KaimingUniform()))
        self.conv3 = nn.Conv2D(
            dim,
            dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups= dim,
            weight_attr=paddle.ParamAttr(initializer=nn.initializer.KaimingUniform()))

    def forward(self, x):
        B, N, C = x.shape
        xx = x.reshape([0, self.dim, self.h, self.w])
        xx = self.conv1(xx)
        xx = paddle.reshape(xx, shape=[B, N, C])
        xx = self.ln1(xx)
        xx = self.act1(xx)
        xx = xx.reshape([0, self.dim, self.h, self.w])
        xx = self.conv2(xx)
        xx = paddle.reshape(xx, shape=[B, N, C])
        xx = self.ln2(xx)
        xx = self.act2(xx)
        xx = xx.reshape([0, self.dim, self.h, self.w])
        xx = self.conv3(xx)
        xx = paddle.reshape(xx, shape=[B, N, C])
        xx = self.drop_path(xx)

        x = x + xx + self.drop_path(self.adaptlayer(self.mixer(self.norm1(x))))
        x = self.norm2(x)
        x = x + self.drop_path(self.mlp(x))

        return x


class PatchEmbed(nn.Layer):
    """ Image to Patch Embedding
    """

    def __init__(self,
                 img_size=[32, 320],
                 in_channels=3,
                 embed_dim=768,
                 sub_num=2,
                 patch_size=[4, 4],
                 mode='pope'):
        super().__init__()
        num_patches = (img_size[1] // (2 ** sub_num)) * \
                      (img_size[0] // (2 ** sub_num))
        self.img_size = img_size
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.norm = None
        if mode == 'pope':
            if sub_num == 2:
                self.proj = nn.Sequential(
                    ConvLNLayer(
                        in_channels=in_channels,
                        out_channels=embed_dim // 2,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        act=nn.GELU,
                        bias_attr=None),
                    ConvLNLayer(
                        in_channels=embed_dim // 2,
                        out_channels=embed_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        act=nn.GELU,
                        bias_attr=None))
            if sub_num == 3:
                self.proj = nn.Sequential(
                    ConvLNLayer(
                        in_channels=in_channels,
                        out_channels=embed_dim // 4,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        act=nn.GELU,
                        bias_attr=None),
                    ConvLNLayer(
                        in_channels=embed_dim // 4,
                        out_channels=embed_dim // 2,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        act=nn.GELU,
                        bias_attr=None),
                    ConvLNLayer(
                        in_channels=embed_dim // 2,
                        out_channels=embed_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        act=nn.GELU,
                        bias_attr=None))
        elif mode == 'linear':
            self.proj = nn.Conv2D(
                1, embed_dim, kernel_size=patch_size, stride=patch_size)
            self.num_patches = img_size[0] // patch_size[0] * img_size[
                1] // patch_size[1]

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose((0, 2, 1))
        return x

class SubSample(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 types='Pool',
                 stride=[2, 1],
                 sub_norm='nn.LayerNorm',
                 act=None):
        super().__init__()
        self.types = types
        if types == 'Pool':
            self.avgpool = nn.AvgPool2D(
                kernel_size=[3, 5], stride=stride, padding=[1, 2])
            self.maxpool = nn.MaxPool2D(
                kernel_size=[3, 5], stride=stride, padding=[1, 2])
            self.proj = nn.Linear(in_channels, out_channels)
        else:
            self.conv = nn.Conv2D(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                weight_attr=paddle.ParamAttr(initializer=nn.initializer.KaimingUniform()))
        self.norm = eval(sub_norm)(out_channels)
        if act is not None:
            self.act = act()
        else:
            self.act = None

    def forward(self, x):

        if self.types == 'Pool':
            x1 = self.avgpool(x)
            x2 = self.maxpool(x)
            x = (x1 + x2) * 0.5
            out = self.proj(x.flatten(2).transpose((0, 2, 1)))
        else:
            x = self.conv(x)
            out = x.flatten(2).transpose((0, 2, 1))
        out = self.norm(out)
        if self.act is not None:
            out = self.act(out)

        return out


class SVTRNet(nn.Layer):
    def __init__(
            self,
            img_size=[32, 320],
            in_channels=3,
            embed_dim=[192, 256, 512],
            depth=[3, 9, 9],
            num_heads=[6, 8, 16],
            mixer=['Local'] * 10 + ['Global'] * 11,  # Local atten, Global atten, Conv
            local_mixer=[[7, 11], [7, 11], [7, 11]],
            patch_merging='Conv',  # Conv, Pool, None
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            last_drop=0.1,
            attn_drop_rate=0.,
            drop_path_rate=0.1,
            norm_layer='nn.LayerNorm',
            sub_norm='nn.LayerNorm',
            epsilon=1e-6,
            out_channels=384,
            out_char_num=40,
            block_unit='Block',
            act='nn.GELU',
            last_stage=True,
            sub_num=2,
            prenorm=True,
            use_lenhead=False,
            **kwargs):
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.prenorm = prenorm
        patch_merging = None if patch_merging != 'Conv' and patch_merging != 'Pool' else patch_merging
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            in_channels=in_channels,
            embed_dim=embed_dim[0],
            sub_num=sub_num)
        num_patches = self.patch_embed.num_patches
        self.HW = [img_size[0] // (2**sub_num), img_size[1] // (2**sub_num)]
        self.pos_embed = self.create_parameter(
            shape=[1, num_patches, embed_dim[0]], default_initializer=zeros_)
        self.add_parameter("pos_embed", self.pos_embed)
        self.pos_drop = nn.Dropout(p=drop_rate)
        Block_unit = eval(block_unit)
        dpr = np.linspace(0, drop_path_rate, sum(depth))

        self.pos_att1 = PositionAttention(max_length=self.HW[0] * self.HW[1],
                                     in_channels=embed_dim[0],
                                     num_channels=192,
                                     h=self.HW[0],
                                     w=self.HW[1])
        self.pos_att2 = PositionAttention(max_length=self.HW[0] * self.HW[1],
                                     in_channels=embed_dim[0],
                                     num_channels=192,
                                     h=self.HW[0],
                                     w=self.HW[1])
        self.pos_mlp = Mlp(in_features=embed_dim[0],
                           hidden_features=embed_dim[0] * mlp_ratio)
        self.pos_norm1 = nn.LayerNorm(embed_dim[0])
        self.pos_norm2 = nn.LayerNorm(embed_dim[0])
        self.pos_act1 = nn.Hardswish()
        self.pos_act2 = nn.Hardswish()
        self.licpa1 = nn.Linear(self.HW[0] * self.HW[1], embed_dim[0])
        self.licpa2 = nn.Linear(self.HW[0] * self.HW[1], embed_dim[0])
        self.linear_to_blk1 = nn.Linear(embed_dim[0], embed_dim[0])

        self.blocks1 = nn.LayerList([
            Block_unit(
                dim=embed_dim[0],
                num_heads=num_heads[0],
                mixer=mixer[0:depth[0]][i],
                HW=self.HW,
                local_mixer=local_mixer[0],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                act_layer=eval(act),
                attn_drop=attn_drop_rate,
                drop_path=dpr[0:depth[0]][i],
                norm_layer=norm_layer,
                epsilon=epsilon,
                prenorm=prenorm,
                H=self.HW[0],
                W=self.HW[1]) for i in range(depth[0])
        ])
        if patch_merging is not None:
            self.sub_sample1 = SubSample(
                embed_dim[0],
                embed_dim[1],
                sub_norm=sub_norm,
                stride=[2, 1],
                types=patch_merging)
            HW = [self.HW[0] // 2, self.HW[1]]
        else:
            HW = self.HW
        self.patch_merging = patch_merging
        self.blocks2 = nn.LayerList([
            Block_unit(
                dim=embed_dim[1],
                num_heads=num_heads[1],
                mixer=mixer[depth[0]:depth[0] + depth[1]][i],
                HW=HW,
                local_mixer=local_mixer[1],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                act_layer=eval(act),
                attn_drop=attn_drop_rate,
                drop_path=dpr[depth[0]:depth[0] + depth[1]][i],
                norm_layer=norm_layer,
                epsilon=epsilon,
                prenorm=prenorm,
                H=self.HW[0] // 2,
                W=self.HW[1]) for i in range(depth[1])
        ])
        if patch_merging is not None:
            self.sub_sample2 = SubSample(
                embed_dim[1],
                embed_dim[2],
                sub_norm=sub_norm,
                stride=[2, 1],
                types=patch_merging)
            HW = [self.HW[0] // 4, self.HW[1]]
        else:
            HW = self.HW
        HW = [self.HW[0] // 4, self.HW[1]]
        self.blocks3 = nn.LayerList([
            Block_unit(
                dim=embed_dim[2],
                num_heads=num_heads[2],
                mixer=mixer[depth[0] + depth[1]:][i],
                HW=HW,
                local_mixer=local_mixer[2],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                act_layer=eval(act),
                attn_drop=attn_drop_rate,
                drop_path=dpr[depth[0] + depth[1]:][i],
                norm_layer=norm_layer,
                epsilon=epsilon,
                prenorm=prenorm,
                H=self.HW[0] // 4,
                W=self.HW[1]) for i in range(depth[2])
        ])
        self.last_stage = last_stage
        if last_stage:
            self.avg_pool = nn.AdaptiveAvgPool2D([1, out_char_num])
            self.last_conv = nn.Conv2D(
                in_channels=embed_dim[2],
                out_channels=self.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                weight_attr=paddle.ParamAttr(initializer=nn.initializer.KaimingUniform()),
                bias_attr=False)
            self.hardswish = nn.Hardswish()
            self.dropout = nn.Dropout(p=last_drop, mode="downscale_in_infer")

        if not prenorm:
            self.norm = eval(norm_layer)(embed_dim[-1], epsilon=epsilon)
        self.use_lenhead = use_lenhead
        if use_lenhead:
            self.len_conv = nn.Linear(embed_dim[2], self.out_channels)
            self.hardswish_len = nn.Hardswish()
            self.dropout_len = nn.Dropout(
                p=last_drop, mode="downscale_in_infer")
        trunc_normal_(self.pos_embed)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed

        xx = x
        vec, xx = self.pos_att1(xx)
        xx = self.licpa1(xx)
        x = x + xx
        x = self.pos_norm1(x)
        x = self.pos_mlp(x)
        x = self.pos_act1(x)

        vec, xx = self.pos_att2(x, vec)
        vec = self.linear_to_blk1(vec)
        xx = self.licpa2(xx)
        x = x + xx
        x = self.pos_norm2(x)
        x = vec + x
        x = self.pos_act2(x)
        x = self.pos_drop(x)

        for blk in self.blocks1:
            x = blk(x)
        if self.patch_merging is not None:
            x = self.sub_sample1(
                x.transpose([0, 2, 1]).reshape(
                    [0, self.embed_dim[0], self.HW[0], self.HW[1]]))
        for blk in self.blocks2:
            x = blk(x)
        if self.patch_merging is not None:
            x = self.sub_sample2(
                x.transpose([0, 2, 1]).reshape(
                    [0, self.embed_dim[1], self.HW[0] // 2, self.HW[1]]))
        for blk in self.blocks3:
            x = blk(x)
        if not self.prenorm:
            x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        if self.use_lenhead:
            len_x = self.len_conv(x.mean(1))
            len_x = self.dropout_len(self.hardswish_len(len_x))
        if self.last_stage:
            if self.patch_merging is not None:
                h = self.HW[0] // 4
            else:
                h = self.HW[0]
            x = self.avg_pool(
                x.transpose([0, 2, 1]).reshape(
                    [0, self.embed_dim[2], h, self.HW[1]]))
            x = self.last_conv(x)
            x = self.hardswish(x)
            x = self.dropout(x)

        if self.use_lenhead:
            return x, len_x
        return x


# x = paddle.Tensor(np.random.rand(32, 3, 32, 320).astype(np.float32))
# model = SVTRNet()
#
# import numpy as np
#
# def calculate_params(model):
#     n_train = 0
#     n_non_train = 0
#     for p in model.parameters():
#         if p.trainable:
#             n_train += np.prod(p.shape)
#         else:
#             print(p.name)
#             n_non_train += np.prod(p.shape)
#     print(n_train + n_non_train, n_train, n_non_train)
#
# calculate_params(model) #42122448 42122448 0
#
# import time
# # 记录开始时间
# start_time = time.time()
# print(model(x).shape)
# end_time = time.time()
# execution_time = end_time - start_time
# print("代码执行时间：{}秒".format(execution_time))