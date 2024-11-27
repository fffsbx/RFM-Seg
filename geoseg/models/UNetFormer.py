import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import timm
from tools.remoteclip_7 import *
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU6()
        )


class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class SeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super(SeparableConv, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GlobalLocalAttention(nn.Module):
    def __init__(self,
                 dim=256,
                 num_heads=16,
                 qkv_bias=False,
                 window_size=8,
                 relative_pos_embedding=True
                 ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.ws = window_size

        self.qkv = Conv(dim, 3*dim, kernel_size=1, bias=qkv_bias)
        self.local1 = ConvBN(dim, dim, kernel_size=3)
        self.local2 = ConvBN(dim, dim, kernel_size=1)
        self.proj = SeparableConvBN(dim, dim, kernel_size=window_size)

        self.attn_x = nn.AvgPool2d(kernel_size=(window_size, 1), stride=1,  padding=(window_size//2 - 1, 0))
        self.attn_y = nn.AvgPool2d(kernel_size=(1, window_size), stride=1, padding=(0, window_size//2 - 1))

        self.relative_pos_embedding = relative_pos_embedding

        if self.relative_pos_embedding:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.ws)
            coords_w = torch.arange(self.ws)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.ws - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.ws - 1
            relative_coords[:, :, 0] *= 2 * self.ws - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

            trunc_normal_(self.relative_position_bias_table, std=.02)

    def pad(self, x, ps):
        _, _, H, W = x.size()
        if W % ps != 0:
            x = F.pad(x, (0, ps - W % ps), mode='reflect')
        if H % ps != 0:
            x = F.pad(x, (0, 0, 0, ps - H % ps), mode='reflect')
        return x

    def pad_out(self, x):
        x = F.pad(x, pad=(0, 1, 0, 1), mode='reflect')
        return x

    def forward(self, x):
        B, C, H, W = x.shape

        local = self.local2(x) + self.local1(x)

        x = self.pad(x, self.ws)
        B, C, Hp, Wp = x.shape
        qkv = self.qkv(x)

        q, k, v = rearrange(qkv, 'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d', h=self.num_heads,
                            d=C//self.num_heads, hh=Hp//self.ws, ww=Wp//self.ws, qkv=3, ws1=self.ws, ws2=self.ws)

        dots = (q @ k.transpose(-2, -1)) * self.scale

        if self.relative_pos_embedding:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.ws * self.ws, self.ws * self.ws, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            dots += relative_position_bias.unsqueeze(0)

        attn = dots.softmax(dim=-1)
        attn = attn @ v

        attn = rearrange(attn, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', h=self.num_heads,
                         d=C//self.num_heads, hh=Hp//self.ws, ww=Wp//self.ws, ws1=self.ws, ws2=self.ws)

        attn = attn[:, :, :H, :W]

        out = self.attn_x(F.pad(attn, pad=(0, 0, 0, 1), mode='reflect')) + \
              self.attn_y(F.pad(attn, pad=(0, 1, 0, 0), mode='reflect'))

        out = out + local
        out = self.pad_out(out)
        out = self.proj(out)
        # print(out.size())
        out = out[:, :, :H, :W]

        return out


class Block(nn.Module):
    def __init__(self, dim=256, num_heads=16,  mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, window_size=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = GlobalLocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, window_size=window_size)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):

        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class WF(nn.Module):
    def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
        super(WF, self).__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        return x


class FeatureRefinementHead(nn.Module):
    def __init__(self, in_channels=64, decode_channels=64):
        super().__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = 1e-8
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

        self.pa = nn.Sequential(nn.Conv2d(decode_channels, decode_channels, kernel_size=3, padding=1, groups=decode_channels),
                                nn.Sigmoid())
        self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                Conv(decode_channels, decode_channels//16, kernel_size=1),
                                nn.ReLU6(),
                                Conv(decode_channels//16, decode_channels, kernel_size=1),
                                nn.Sigmoid())

        self.shortcut = ConvBN(decode_channels, decode_channels, kernel_size=1)
        self.proj = SeparableConvBN(decode_channels, decode_channels, kernel_size=3)
        self.act = nn.ReLU6()

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        shortcut = self.shortcut(x)
        pa = self.pa(x) * x
        ca = self.ca(x) * x
        x = pa + ca
        x = self.proj(x) + shortcut
        x = self.act(x)

        return x


class AuxHead(nn.Module):

    def __init__(self, in_channels=64, num_classes=8):
        super().__init__()
        self.conv = ConvBNReLU(in_channels, in_channels)
        self.drop = nn.Dropout(0.1)
        self.conv_out = Conv(in_channels, num_classes, kernel_size=1)

    def forward(self, x, h, w):
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False)
        return feat


class Decoder(nn.Module):
    #encoder_channels（编码器通道数）、decode_channels（解码通道数）、dropout（dropout 比率）、window_size（窗口大小）和 num_classes（类别数）
    def __init__(self,encoder_channels=(64, 128, 256, 512),decode_channels=64,dropout=0.1,window_size=8,num_classes=6):
        super(Decoder, self).__init__()
        #定义一个 1x1 卷积并进行批归一化的模块，输入通道数为编码器最后一层的通道数，输出通道数为解码器的通道数
        self.pre_conv = ConvBN(encoder_channels[-1], decode_channels, kernel_size=1)
        #定义一个自定义的块模块，输入通道数为解码器通道数，头数为 8，窗口大小为给定的大小
        self.b4 = Block(dim=decode_channels, num_heads=8, window_size=window_size)
        self.b3 = Block(dim=decode_channels, num_heads=8, window_size=window_size)
        self.p3 = WF(encoder_channels[-2], decode_channels)
        self.b2 = Block(dim=decode_channels, num_heads=8, window_size=window_size)
        self.p2 = WF(encoder_channels[-3], decode_channels)
        if self.training:
            #定义上采样模块，使用双线性插值上采样，缩放因子为 4 和 2
            self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)
            self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)
            ## 定义辅助头模块，输入通道数为解码器通道数，类别数为给定的类别数
            self.aux_head = AuxHead(decode_channels, num_classes)
        #定义特征细化头模块，输入通道数为编码器倒数第二层的通道数，输出通道数为解码器通道数
        self.p1 = FeatureRefinementHead(encoder_channels[-4], decode_channels)
        # 定义分割头模块，包含一系列卷积、批归一化和激活函数，最后输出类别数的通道数
        self.segmentation_head = nn.Sequential(ConvBNReLU(decode_channels, decode_channels),
                                               nn.Dropout2d(p=dropout, inplace=True),
                                               Conv(decode_channels, num_classes, kernel_size=1))
        # self.a4 = nn.Conv2d(in_channels=576, out_channels=64, kernel_size=3, padding=1)
        # self.a3 = nn.Conv2d(in_channels=320, out_channels=64, kernel_size=3, padding=1)
        # self.a2 = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=3, padding=1)
        self.a1 = nn.Conv2d(in_channels=70, out_channels=64, kernel_size=3, padding=1)

        # 初始化模型权重
        self.init_weight()
    def forward(self, res1, res2, res3, res4, h, w):
        if self.training:
            # 对输入的 res4 应用预卷积和块模块 b4，并将结果上采样
            x = self.pre_conv(res4)#torch.Size([16, 64, 16, 16])
            # x = torch.cat((x, res4), dim=1)
            # x = self.a4(x)
            x = self.b4(x)#torch.Size([16, 64, 16, 16])
            h4 = self.up4(x)#torch.Size([16, 64, 64, 64])
            # 对上采样后的结果应用特征金字塔模块 p3，并将结果上采样
            x = self.p3(x, res3)#torch.Size([16, 64, 32, 32])
            # x = torch.cat((x, res3), dim=1)
            # x = self.a3(x)
            x = self.b3(x)#torch.Size([16, 64, 32, 32])
            h3 = self.up3(x)#torch.Size([16, 64, 64, 64])
            # 对上采样后的结果应用特征金字塔模块 p2
            x = self.p2(x, res2)#torch.Size([16, 64, 64, 64])
            # x = torch.cat((x, res2), dim=1)
            # x = self.a2(x)
            x = self.b2(x)#torch.Size([16, 64, 64, 64])
            h2 = x#torch.Size([16, 64, 64, 64])
            # 对上采样后的结果应用特征细化头模块 p1
            x = self.p1(x, res1)#torch.Size([16, 64, 128, 128])
            # x = torch.cat((x, res1), dim=1)
            # x = self.a1(x)
            # 将结果输入分割头模块进行分割，得到分割结果
            x = self.segmentation_head(x)#torch.Size([16, 6, 128, 128])
            # 将分割结果通过双线性插值上采样到指定的大小
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)#torch.Size([16, 6, 512, 512])
            # 将上采样后的结果进行融合，并通过辅助头模块得到辅助的分割结果
            ah = h4 + h3 + h2#torch.Size([16, 64, 64, 64])
            ah = self.aux_head(ah, h, w)#torch.Size([16, 6, 512, 512])
            # 返回主分割结果和辅助分割结果
            return x, ah
        else:
            x = self.pre_conv(res4)#torch.Size([16, 64, 32, 32])
            x = self.b4(x)#torch.Size([16, 64, 32, 32])
            # x = torch.cat((x, res4), dim=1)
            # x = self.a4(x)

            x = self.p3(x, res3)#torch.Size([16, 64, 64, 64])
            x = self.b3(x)#torch.Size([16, 64, 64, 64])
            # x = torch.cat((x, res3), dim=1)
            # x = self.a3(x)

            x = self.p2(x, res2)#torch.Size([16, 64, 128, 128])
            x = self.b2(x)#torch.Size([16, 64, 128, 128])
            # x = torch.cat((x, res2), dim=1)
            # x = self.a2(x)

            x = self.p1(x, res1)#torch.Size([16, 64, 256, 256])
            # x = torch.cat((x, res1), dim=1)
            # x = self.a1(x)
            x = self.segmentation_head(x)#torch.Size([16, 6, 256, 256])
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)#torch.Size([16, 6, 1024, 1024])
            return x

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class RemoteClip(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self):
        model_name = 'RN50'  # 'RN50' or 'ViT-B-32' or 'ViT-L-14'
        model, _, preprocess = open_clip.create_model_and_transforms(model_name)
        path_to_your_checkpoints = 'checkpoints/models--chendelong--RemoteCLIP'
        ckpt = torch.load(f"{path_to_your_checkpoints}/RemoteCLIP-{model_name}.pt", map_location="cuda")
        message = model.load_state_dict(ckpt)
        print(message)
        model = model.cuda().eval()
        tokenizer = open_clip.get_tokenizer(model_name)
        text_queries = [
            "nothing",
            'road',
            'road next to the low vegetation',
            'road next to the Red circular tree',
            'road next to the house',
            'road next to the car',
            'road next to the water',
            'road next to a lake',

            'building',
            "building next to the circular tree",
            'building next to the water',
            'building next to the building',
            'building next to the road',
            'building next to the red low vegetation',
            'building next to a lake',

            'red low vegetation',
            'red low vegetation next to the red circular tree',
            'red low vegetation next to the building',
            'red low vegetation next to the car',
            'red low vegetation next to the water',
            'red low vegetation next to the road',
            'red low vegetation next to a lake',

            'red circular tree',
            'red circular tree next to the building',
            'red circular tree next to the road',
            'red circular tree next to the water',
            'red circular tree next to the car',
            'red circular tree on the low vegetation',
            'red circular tree next to a lake',

            'car'
            'car on the low vegetation',
            "car next to the water",
            "car next to a lake",
            "car next to the building",
            "car next to the road",
            "car next to the red circular tree",

            'lake',
            "water",
            "water next to Red low vegetation",
            "water next to the building",
            "water next to the road",
            "water next to red circular tree",
            'water next to the car',
        ]
        # text_queries = [
        #
        #
        #     'building',
        #     "building next to the trees",
        #     'building next to a lake',
        #     'building next to the building',
        #     'building next to the road',
        #
        #     'road',
        #     'road next to the low vegetation',
        #     'road next to the trees',
        #     'road next to the house',
        #     'car on the low vegetation',
        #     "car next to a lake",
        #     "car next to the building",
        #     "car next to the road",
        #
        #     'low vegetation',
        #     'low vegetation next to the trees',
        #     'low vegetation next to the building',
        #     'low vegetation next to the road',
        #     'tree'
        #     'tree next to the building',
        #     'tree next to the road',
        #     'tree next to a lake',
        #     'tree on the low vegetation',
        #     "water",
        #     "water next to Red low vegetation",
        #     "water next to the building",
        #     "water next to the road",
        #     "water next to red trees",
        #     'background',
        #     "nothing"
        # ]
        # text_queries = ['Building', 'Road', 'Tree', 'LowVeg', 'Moving_Car', 'Static_Car', 'Human', 'Clutter']
        text = tokenizer(text_queries)
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_features = model.encode_text(text.cuda())
        return text_features

# class FusionAttention(nn.Module):
#     def __init__(self,
#                  dim=256,
#                  ssmdims=256,
#                  num_heads=16,
#                  qkv_bias=False,
#                  window_size=8,
#                  relative_pos_embedding=True
#                  ):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // self.num_heads
#         self.scale = head_dim ** -0.5
#         self.ws = window_size
#
#         self.qkv = Conv(dim, 3*dim, kernel_size=1, bias=qkv_bias)
#         self.local1 = ConvBN(ssmdims, dim, kernel_size=3)
#         self.local2 = ConvBN(ssmdims, dim, kernel_size=1)
#         self.proj = SeparableConvBN(dim, dim, kernel_size=window_size)
#
#         self.attn_x = nn.AvgPool2d(kernel_size=(window_size, 1), stride=1,  padding=(window_size//2 - 1, 0))
#         self.attn_y = nn.AvgPool2d(kernel_size=(1, window_size), stride=1, padding=(0, window_size//2 - 1))
#
#         self.relative_pos_embedding = relative_pos_embedding
#
#         if self.relative_pos_embedding:
#             # define a parameter table of relative position bias
#             self.relative_position_bias_table = nn.Parameter(
#                 torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
#
#             # get pair-wise relative position index for each token inside the window
#             coords_h = torch.arange(self.ws)
#             coords_w = torch.arange(self.ws)
#             coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
#             coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
#             relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
#             relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
#             relative_coords[:, :, 0] += self.ws - 1  # shift to start from 0
#             relative_coords[:, :, 1] += self.ws - 1
#             relative_coords[:, :, 0] *= 2 * self.ws - 1
#             relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
#             self.register_buffer("relative_position_index", relative_position_index)
#
#             trunc_normal_(self.relative_position_bias_table, std=.02)
#
#     def pad(self, x, ps):
#         _, _, H, W = x.size()
#         if W % ps != 0:
#             x = F.pad(x, (0, ps - W % ps), mode='reflect')
#         if H % ps != 0:
#             x = F.pad(x, (0, 0, 0, ps - H % ps), mode='reflect')
#         return x
#
#     def pad_out(self, x):
#         x = F.pad(x, pad=(0, 1, 0, 1), mode='reflect')
#         return x
#
#     def forward(self, x, y):
#         ## x from res, need global; y from smm, need local
#         B, C, H, W = x.shape
#
#         local = self.local2(y) + self.local1(y)
#
#         x = self.pad(x, self.ws)
#         B, C, Hp, Wp = x.shape
#         qkv = self.qkv(x)
#
#         q, k, v = rearrange(qkv, 'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d', h=self.num_heads,
#                             d=C//self.num_heads, hh=Hp//self.ws, ww=Wp//self.ws, qkv=3, ws1=self.ws, ws2=self.ws)
#
#         dots = (q @ k.transpose(-2, -1)) * self.scale
#
#         if self.relative_pos_embedding:
#             relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
#                 self.ws * self.ws, self.ws * self.ws, -1)  # Wh*Ww,Wh*Ww,nH
#             relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
#             dots += relative_position_bias.unsqueeze(0)
#
#         attn = dots.softmax(dim=-1)
#         attn = attn @ v
#
#         attn = rearrange(attn, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', h=self.num_heads,
#                          d=C//self.num_heads, hh=Hp//self.ws, ww=Wp//self.ws, ws1=self.ws, ws2=self.ws)
#
#         attn = attn[:, :, :H, :W]
#
#         out = self.attn_x(F.pad(attn, pad=(0, 0, 0, 1), mode='reflect')) + \
#               self.attn_y(F.pad(attn, pad=(0, 1, 0, 0), mode='reflect'))
#
#         out = out + local
#         out = self.pad_out(out)
#         out = self.proj(out)
#         # print(out.size())
#         out = out[:, :, :H, :W]
#
#         return out

class FusionAttention(nn.Module):
    def __init__(self,
                 dim=256,
                 ssmdims=256,
                 num_heads=16,
                 qkv_bias=False,
                 window_size=8,
                 relative_pos_embedding=True
                 ):
        super().__init__()

        self.proj = SeparableConvBN(dim, dim, kernel_size=window_size)


    def pad_out(self, x):
        x = F.pad(x, pad=(0, 1, 0, 1), mode='reflect')
        return x

    def forward(self, x, y):
        B, C, H, W = x.shape
        out = x + y
        out = self.pad_out(out)
        out = self.proj(out)
        # print(out.size())
        out = out[:, :, :H, :W]

        return out

class Mlp_attention(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0.):
        super().__init__()
        out_features = out_features or in_features

        self.fc2 = nn.Conv2d(in_features, out_features, 1, 1, 0, bias=True)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):

        x = self.fc2(x)
        x = self.drop(x)
        return x

class FusionTwo(nn.Module):
    def __init__(self, dim=512, ssmdims=512, num_heads=16,  mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, window_size=8):
        super().__init__()
        self.normx = norm_layer(dim)
        self.normy = norm_layer(ssmdims)
        self.attn = FusionAttention(dim, ssmdims, num_heads=num_heads, qkv_bias=qkv_bias, window_size=window_size)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_attention(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x, y):
        x = x + self.drop_path(self.attn(self.normx(x), self.normy(y)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class FusionThree(nn.Module):
    def __init__(self, dim=512, ssmdims=2048, num_heads=16, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, window_size=8):
        super().__init__()
        self.normx1 = norm_layer(dim)
        self.normy = norm_layer(dim)
        self.normz = norm_layer(ssmdims)  # 新增的规范化层用于z
        self.normx2 = norm_layer(dim)
        self.attnxy = FusionAttention(dim, dim, num_heads=num_heads, qkv_bias=qkv_bias, window_size=window_size)
        self.attnxz = FusionAttention(dim, ssmdims, num_heads=num_heads, qkv_bias=qkv_bias, window_size=window_size)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_attention(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
                       drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x, y, z):  # 修改 forward 方法的参数列表
        x = x + 0.5*self.drop_path(self.attnxy(self.normx1(x), self.normy(y))) + 0.5*self.drop_path(self.attnxz(self.normx2(x), self.normz(z)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class UNetFormer(nn.Module):
    def __init__(self,
                 decode_channels=64,
                 dropout=0.1,
                 backbone_name='swsl_resnet18',
                 pretrained=True,
                 window_size=8,
                 num_classes=6
                 ):
        super().__init__()
        model_weights_path = '/home/ligong1/zqy_project/GeoSeg_clip/model_weights/potsdam/unetformer-r18-768crop-ms-e45/unetformer-r18-768crop-ms-e45.ckpt'
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 加载整个模型的权重
        state_dict = torch.load(model_weights_path, map_location=device)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        self.backbone = timm.create_model(backbone_name, features_only=True, output_stride=32,out_indices=(1, 2, 3, 4), pretrained=pretrained)
        # 提取码器部分的权重
        backbone_state_dict = {k.replace('net.backbone.', ''): v for k, v in state_dict.items() if
                              k.startswith('net.backbone.')}

        self.backbone.load_state_dict(backbone_state_dict, strict=True)
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.remotrclip_image = remoteclip_image()
        self.remotrclip_text = remoteclip_text()
        encoder_channels = self.backbone.feature_info.channels()
        self.decoder = Decoder(encoder_channels, decode_channels, dropout, window_size, num_classes)
        # 提取解码器部分的权重
        decoder_state_dict = {k.replace('net.decoder.', ''): v for k, v in state_dict.items() if
                              k.startswith('net.decoder.')}

        # 注入解码器部分的权重
        self.decoder.load_state_dict(decoder_state_dict, strict=True)
        for param in self.decoder.parameters():
             param.requires_grad = False
                # self.conv_layer1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
                # self.conv_layer2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
                # self.conv_layer3 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1)
                # self.conv_layer4 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, padding=1)
                # self.conv_layer5 = nn.Conv2d(in_channels=2560, out_channels=512, kernel_size=3, padding=1)
        self.text_clip = RemoteClip()
        self.Fuse1 = FusionTwo(64, 64)
        self.Fuse2 = FusionTwo(128, 128)
        self.Fuse3 = FusionTwo(256, 256)
        self.Fuse4 = FusionThree(512,512)
    def forward(self, x,batch_idx=0):
        h, w = x.size()[-2:]
        res1, res2, res3, res4 = self.backbone(x)
        text_features = self.text_clip()
        result_tensor = self.remotrclip_text(x,text_features)
        clip1,clip2,clip3,clip4= self.remotrclip_image(x)
        # # 添加一个新的维度
        res1 = self.Fuse1(res1.cuda().to(torch.float), clip1.cuda().to(torch.float))
        res2 = self.Fuse2(res2.cuda().to(torch.float), clip2.cuda().to(torch.float))
        res3 = self.Fuse3(res3.cuda().to(torch.float), clip3.cuda().to(torch.float))
        res4 = self.Fuse4(res4.cuda().to(torch.float), clip4.cuda().to(torch.float),result_tensor.cuda().to(torch.float))
        if self.training:
            x, ah = self.decoder(res1.to(torch.float), res2.to(torch.float), res3.to(torch.float), res4.to(torch.float), h, w)
            return x, ah
        else:
            x = self.decoder(res1.to(torch.float), res2.to(torch.float), res3.to(torch.float), res4.to(torch.float), h, w)
            return x
