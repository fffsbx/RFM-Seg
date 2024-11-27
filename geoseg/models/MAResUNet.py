import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from torch.nn import Module, Conv2d, Parameter, Softmax

from functools import partial
from functools import partial
from einops import rearrange, repeat

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import timm
from tools.remoteclip_7 import *

nonlinearity = partial(F.relu, inplace=True)


def conv3otherRelu(in_planes, out_planes, kernel_size=None, stride=None, padding=None):
    # 3x3 convolution with padding and relu
    if kernel_size is None:
        kernel_size = 3
    assert isinstance(kernel_size, (int, tuple)), 'kernel_size is not in (int, tuple)!'

    if stride is None:
        stride = 1
    assert isinstance(stride, (int, tuple)), 'stride is not in (int, tuple)!'

    if padding is None:
        padding = 1
    assert isinstance(padding, (int, tuple)), 'padding is not in (int, tuple)!'

    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
        nn.ReLU(inplace=True)  # inplace=True
    )


def l2_norm(x):
    return torch.einsum("bcn, bn->bcn", x, 1 / torch.norm(x, p=2, dim=-2))


class PAM_Module(Module):
    def __init__(self, in_places, scale=8, eps=1e-6):
        super(PAM_Module, self).__init__()
        self.gamma = Parameter(torch.zeros(1))
        self.in_places = in_places
        # self.exp_feature = exp_feature_map
        # self.tanh_feature = tanh_feature_map
        self.l2_norm = l2_norm
        self.eps = eps

        self.query_conv = Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_places, out_channels=in_places, kernel_size=1)

    def forward(self, x):
        # Apply the feature map to the queries and keys
        batch_size, chnnels, width, height = x.shape
        Q = self.query_conv(x).view(batch_size, -1, width * height)
        K = self.key_conv(x).view(batch_size, -1, width * height)
        V = self.value_conv(x).view(batch_size, -1, width * height)

        Q = self.l2_norm(Q).permute(-3, -1, -2)
        K = self.l2_norm(K)

        tailor_sum = 1 / (width * height + torch.einsum("bnc, bc->bn", Q, torch.sum(K, dim=-1) + self.eps))
        value_sum = torch.einsum("bcn->bc", V).unsqueeze(-1)
        value_sum = value_sum.expand(-1, chnnels, width * height)

        matrix = torch.einsum('bmn, bcn->bmc', K, V)
        matrix_sum = value_sum + torch.einsum("bnm, bmc->bcn", Q, matrix)

        weight_value = torch.einsum("bcn, bn->bcn", matrix_sum, tailor_sum)
        weight_value = weight_value.view(batch_size, chnnels, height, width)

        return (x + self.gamma * weight_value).contiguous()


class CAM_Module(Module):
    def __init__(self):
        super(CAM_Module, self).__init__()
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        batch_size, chnnels, width, height = x.shape
        proj_query = x.view(batch_size, chnnels, -1)
        proj_key = x.view(batch_size, chnnels, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(batch_size, chnnels, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(batch_size, chnnels, height, width)

        out = self.gamma * out + x
        return out


class PAM_CAM_Layer(nn.Module):
    def __init__(self, in_ch):
        super(PAM_CAM_Layer, self).__init__()
        self.conv1 = conv3otherRelu(in_ch, in_ch)

        self.PAM = PAM_Module(in_ch)
        self.CAM = CAM_Module()

        self.conv2P = nn.Sequential(nn.Dropout2d(0.1, False), conv3otherRelu(in_ch, in_ch, 1, 1, 0))
        self.conv2C = nn.Sequential(nn.Dropout2d(0.1, False), conv3otherRelu(in_ch, in_ch, 1, 1, 0))

        self.conv3 = nn.Sequential(nn.Dropout2d(0.1, False), conv3otherRelu(in_ch, in_ch, 1, 1, 0))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2P(self.PAM(x)) + self.conv2C(self.CAM(x))
        return self.conv3(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x

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
            'road next to the low vegetation',
            'road next to the trees',
            'road next to the house',

            'building',
            "building next to the trees",
            'building next to a lake',
            'building next to the building',
            'building next to the road',

            'Red low vegetation',
            'Red low vegetation next to the trees',
            'Red low vegetation next to the building',
            'Red low vegetation next to the road',

            'Red circular tree next to the building',
            'Red circular tree next to the road',
            'Red circular tree next to a lake',
            'Red circular tree on the low vegetation',

            'car on the low vegetation',
            "car next to a lake",
            "car next to the building",
            "car next to the road",

            "water",
            "water next to Red low vegetation",
            "water next to the building",
            "water next to the road",
            "water next to red trees",
        ]
        text = tokenizer(text_queries)
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_features = model.encode_text(text.cuda())
        return text_features

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
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.ws = window_size

        self.qkv = Conv(dim, 3*dim, kernel_size=1, bias=qkv_bias)
        self.local1 = ConvBN(ssmdims, dim, kernel_size=3)
        self.local2 = ConvBN(ssmdims, dim, kernel_size=1)
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

    def forward(self, x, y):
        ## x from res, need global; y from smm, need local
        B, C, H, W = x.shape

        local = self.local2(y) + self.local1(y)

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


class FusionTwo(nn.Module):
    def __init__(self, dim=512, ssmdims=512, num_heads=16,  mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, window_size=8):
        super().__init__()
        self.normx = norm_layer(dim)
        self.normy = norm_layer(ssmdims)
        self.attn = FusionAttention(dim, ssmdims, num_heads=num_heads, qkv_bias=qkv_bias, window_size=window_size)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=drop)
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
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
                       drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x, y, z):  # 修改 forward 方法的参数列表
        x = x + self.drop_path(self.attnxy(self.normx1(x), self.normy(y))) + self.drop_path(self.attnxz(self.normx2(x), self.normz(z)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class MAResUNet(nn.Module):
    def __init__(self, num_channels=3, num_classes=5):
        super(MAResUNet, self).__init__()
        self.name = 'MAResUNet'

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        model_weights_path = '/home/ligong1/zqy_project/GeoSeg_clip/model_weights/potsdam/maresunet-r18-512-crop-ms-e30/maresunet-r18-512-crop-ms-e30.ckpt'
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 加载整个模型的权重
        state_dict = torch.load(model_weights_path, map_location=device)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        self.firstconv = resnet.conv1
        firstconv_state_dict = {k.replace('net.firstconv.', ''): v for k, v in state_dict.items() if
                               k.startswith('net.firstconv.')}

        self.firstconv.load_state_dict(firstconv_state_dict, strict=True)
        for param in self.firstconv.parameters():
            param.requires_grad = False

        self.firstbn = resnet.bn1
        firstbn_state_dict = {k.replace('net.firstbn.', ''): v for k, v in state_dict.items() if
                               k.startswith('net.firstbn.')}

        self.firstbn.load_state_dict(firstbn_state_dict, strict=True)
        for param in self.firstbn.parameters():
            param.requires_grad = False

        self.firstrelu = resnet.relu
        firstrelu_state_dict = {k.replace('net.firstrelu.', ''): v for k, v in state_dict.items() if
                               k.startswith('net.firstrelu.')}

        self.firstrelu.load_state_dict(firstrelu_state_dict, strict=True)
        for param in self.firstrelu.parameters():
            param.requires_grad = False

        self.firstmaxpool = resnet.maxpool
        firstmaxpool_state_dict = {k.replace('net.firstmaxpool.', ''): v for k, v in state_dict.items() if
                               k.startswith('net.firstmaxpool.')}

        self.firstmaxpool.load_state_dict(firstmaxpool_state_dict, strict=True)
        for param in self.firstmaxpool.parameters():
            param.requires_grad = False

        self.encoder1 = resnet.layer1
        encoder1_state_dict = {k.replace('net.encoder1.', ''): v for k, v in state_dict.items() if
                               k.startswith('net.encoder1.')}

        self.encoder1.load_state_dict(encoder1_state_dict, strict=True)
        for param in self.encoder1.parameters():
            param.requires_grad = False

        self.encoder2 = resnet.layer2
        encoder2_state_dict = {k.replace('net.encoder2.', ''): v for k, v in state_dict.items() if
                               k.startswith('net.encoder2.')}

        self.encoder2.load_state_dict(encoder2_state_dict, strict=True)
        for param in self.encoder2.parameters():
            param.requires_grad = False

        self.encoder3 = resnet.layer3
        encoder3_state_dict = {k.replace('net.encoder3.', ''): v for k, v in state_dict.items() if
                               k.startswith('net.encoder3.')}

        self.encoder3.load_state_dict(encoder3_state_dict, strict=True)
        for param in self.encoder3.parameters():
            param.requires_grad = False

        self.encoder4 = resnet.layer4
        encoder4_state_dict = {k.replace('net.encoder4.', ''): v for k, v in state_dict.items() if
                               k.startswith('net.encoder4.')}

        self.encoder4.load_state_dict(encoder4_state_dict, strict=True)
        for param in self.encoder4.parameters():
            param.requires_grad = False


        self.attention4 = PAM_CAM_Layer(filters[3])
        attention4_state_dict = {k.replace('net.attention4.', ''): v for k, v in state_dict.items() if
                               k.startswith('net.attention4.')}

        self.attention4.load_state_dict(attention4_state_dict, strict=True)
        for param in self.attention4.parameters():
            param.requires_grad = False

        self.attention3 = PAM_CAM_Layer(filters[2])
        attention3_state_dict = {k.replace('net.attention3.', ''): v for k, v in state_dict.items() if
                               k.startswith('net.attention3.')}

        self.attention3.load_state_dict(attention3_state_dict, strict=True)
        for param in self.attention3.parameters():
            param.requires_grad = False

        self.attention2 = PAM_CAM_Layer(filters[1])
        attention2_state_dict = {k.replace('net.attention2.', ''): v for k, v in state_dict.items() if
                               k.startswith('net.attention2.')}

        self.attention2.load_state_dict(attention2_state_dict, strict=True)
        for param in self.attention2.parameters():
            param.requires_grad = False

        self.attention1 = PAM_CAM_Layer(filters[0])
        attention1_state_dict = {k.replace('net.attention1.', ''): v for k, v in state_dict.items() if
                               k.startswith('net.attention1.')}

        self.attention1.load_state_dict(attention1_state_dict, strict=True)
        for param in self.attention1.parameters():
            param.requires_grad = False


        self.decoder4 = DecoderBlock(filters[3], filters[2])
        decoder4_state_dict = {k.replace('net.decoder4.', ''): v for k, v in state_dict.items() if
                               k.startswith('net.decoder4.')}

        self.decoder4.load_state_dict(decoder4_state_dict, strict=True)
        for param in self.decoder4.parameters():
            param.requires_grad = False

        self.decoder3 = DecoderBlock(filters[2], filters[1])
        decoder3_state_dict = {k.replace('net.decoder3.', ''): v for k, v in state_dict.items() if
                               k.startswith('net.decoder3.')}

        self.decoder3.load_state_dict(decoder3_state_dict, strict=True)
        for param in self.decoder3.parameters():
            param.requires_grad = False

        self.decoder2 = DecoderBlock(filters[1], filters[0])
        decoder2_state_dict = {k.replace('net.decoder2.', ''): v for k, v in state_dict.items() if
                               k.startswith('net.decoder2.')}

        self.decoder2.load_state_dict(decoder2_state_dict, strict=True)
        for param in self.decoder2.parameters():
            param.requires_grad = False

        self.decoder1 = DecoderBlock(filters[0], filters[0])
        decoder1_state_dict = {k.replace('net.decoder1.', ''): v for k, v in state_dict.items() if
                               k.startswith('net.decoder1.')}

        self.decoder1.load_state_dict(decoder1_state_dict, strict=True)
        for param in self.decoder1.parameters():
            param.requires_grad = False


        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        finaldeconv1_state_dict = {k.replace('net.finaldeconv1.', ''): v for k, v in state_dict.items() if
                               k.startswith('net.finaldeconv1.')}

        self.finaldeconv1.load_state_dict(finaldeconv1_state_dict, strict=True)
        for param in self.finaldeconv1.parameters():
            param.requires_grad = False

        self.finalrelu1 = nonlinearity

        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        finalconv2_state_dict = {k.replace('net.finalconv2.', ''): v for k, v in state_dict.items() if
                               k.startswith('net.finalconv2.')}

        self.finalconv2.load_state_dict(finalconv2_state_dict, strict=True)
        for param in self.finalconv2.parameters():
            param.requires_grad = False

        self.finalrelu2 = nonlinearity

        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)
        finalconv3_state_dict = {k.replace('net.finalconv3.', ''): v for k, v in state_dict.items() if
                               k.startswith('net.finalconv3.')}

        self.finalconv3.load_state_dict(finalconv3_state_dict, strict=True)
        for param in self.finalconv3.parameters():
            param.requires_grad = False


        self.remotrclip_image = remoteclip_image()
        self.remotrclip_text = remoteclip_text()
        self.text_clip = RemoteClip()
        self.Fuse1 = FusionTwo(64, 64)
        self.Fuse2 = FusionTwo(128, 128)
        self.Fuse3 = FusionTwo(256, 256)
        self.Fuse4 = FusionThree(512,2048)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def forward(self, x):
        # Encoder
        x1 = self.firstconv(x)
        x1 = self.firstbn(x1)
        x1 = self.firstrelu(x1)
        x1 = self.firstmaxpool(x1)
        e1 = self.encoder1(x1)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        e4 = self.attention4(e4)

        text_features = self.text_clip()
        result_tensor = self.remotrclip_text(x,text_features)
        clip1,clip2,clip3,clip4= self.remotrclip_image(x)
        e1 = self.Fuse1(e1.cuda().to(torch.float), clip1.cuda().to(torch.float))
        e2 = self.Fuse2(e2.cuda().to(torch.float), clip2.cuda().to(torch.float))
        e3 = self.Fuse3(e3.cuda().to(torch.float), clip3.cuda().to(torch.float))
        e4 = self.Fuse4(e4.cuda().to(torch.float),clip4.cuda().to(torch.float),result_tensor.cuda().to(torch.float))

        # Decoder
        d4 = self.decoder4(e4) + self.attention3(e3)
        d3 = self.decoder3(d4) + self.attention2(e2)
        d2 = self.decoder2(d3) + self.attention1(e1)
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        # d3 = F.interpolate(d3, size=x.size()[2:], mode='bilinear', align_corners=False)

        # return torch.sigmoid(out)
        return out


if __name__ == '__main__':
    num_classes = 10
    in_batch, inchannel, in_h, in_w = 10, 3, 256, 256
    x = torch.randn(in_batch, inchannel, in_h, in_w)
    net = MAResUNet(3)
    out = net(x)
    print(out.shape)