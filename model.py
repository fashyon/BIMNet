import torch
from torch import nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse
from timm.models.layers import DropPath, to_2tuple

class convd(nn.Module):
    def __init__(self, inputchannel, outchannel, kernel_size, stride):
        super(convd, self).__init__()
        self.padding = nn.ReflectionPad2d(kernel_size // 2)
        self.conv = nn.Sequential(nn.Conv2d(inputchannel, outchannel, kernel_size, stride), nn.LeakyReLU(0.2))

    def forward(self, x):
        x = self.conv(self.padding(x))
        return x

class ConvPosEnc(nn.Module):
    def __init__(self, dim, k=3, act=False, normtype=False):
        super(ConvPosEnc, self).__init__()
        self.proj = nn.Conv2d(dim,
                              dim,
                              to_2tuple(k),
                              to_2tuple(1),
                              to_2tuple(k // 2),
                              groups=dim)
        self.normtype = normtype
        if self.normtype == 'batch':
            self.norm = nn.BatchNorm2d(dim)
        elif self.normtype == 'layer':
            self.norm = nn.LayerNorm(dim)
        self.activation = nn.GELU() if act else nn.Identity()

    def forward(self, x,H,W):
        B, N, C = x.shape
        assert N == H * W

        feat = x.transpose(1, 2).view(B, C, H, W)
        feat = self.proj(feat)
        if self.normtype == 'batch':
            feat = self.norm(feat).flatten(2).transpose(1, 2)
        elif self.normtype == 'layer':
            feat = self.norm(feat.flatten(2).transpose(1, 2))
        else:
            feat = feat.flatten(2).transpose(1, 2)
        x = x + self.activation(feat)
        return x

class Spatial_Adaptive_Fusion_Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.DConv_first = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=3, stride=1,padding=1,groups=dim), nn.LeakyReLU(0.2))
        self.DConv_premul = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=3, stride=1,padding=1,groups=dim), nn.LeakyReLU(0.2))
        self.DConv_postmul = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=3, stride=1,padding=1,groups=dim), nn.LeakyReLU(0.2))
        self.DConv_preadd = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=3, stride=1,padding=1,groups=dim), nn.LeakyReLU(0.2))

    def forward(self,JI,EO):
        B,C,H,W = EO.shape
        _,_,h,w = JI.shape
        JI_first_conv = self.DConv_first(JI)
        JI_conv_premul = self.DConv_premul(JI_first_conv)
        JI_conv_postmul = self.DConv_postmul(EO * JI_conv_premul)
        JO = self.DConv_preadd(JI_first_conv) + JI_conv_postmul + EO
        JI_out = F.interpolate(JI_first_conv, size=[H*2, W*2], mode='bilinear')
        return JI_out,JO

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class Wave_Conv_Self_Attention(nn.Module):
    def __init__(self, dim, window_size, num_heads, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        self.logit_scale_Low = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)
        self.logit_scale_High = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)
        self.query = nn.Linear(dim, dim, bias=True)
        self.KV_Low = nn.Linear(dim, dim * 2, bias=True)
        self.L_conv = nn.Sequential(nn.Conv2d(dim,dim,kernel_size=3,stride=1,padding=1,groups=dim),nn.LeakyReLU(0.2))
        self.H_conv = nn.Sequential(nn.Conv2d(dim,dim,kernel_size=3,stride=1,padding=1,groups=dim),nn.LeakyReLU(0.2))
        self.Rst_conv = nn.Sequential(nn.Conv2d(dim,dim,kernel_size=3,stride=1,padding=1,groups=dim),nn.LeakyReLU(0.2))
        self.KV_High = nn.Linear(dim, dim * 2, bias=True)
        self.attn_drop_L = nn.Dropout(attn_drop)
        self.attn_drop_H = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.DWT = DWTForward(J=1, wave='haar').cuda()
        self.IDWT = DWTInverse(wave='haar').cuda()
        self.fuse_High = convd(3*dim,dim,1,1)
        self.fuse_All = nn.Linear(3*dim, dim)

    def forward(self, x,H,W):
        B_, N, C = x.shape

        # Frequency Division Self-Attention Part (FDSP)
        Q = self.query(x).reshape(B_, N, self.num_heads, -1).permute(0, 2, 1, 3)
        x_ = x.transpose(1,2).view(B_,C,self.window_size[0],self.window_size[1])
        x_LL,x_LH,x_HL,x_HH = self.split_DWT(self.DWT(x_))
        KV_L=self.KV_Low(x_LL.flatten(2).transpose(1,2))
        KV_L = KV_L.reshape(B_, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        KL,VL = KV_L[0],KV_L[1]

        x_H = self.fuse_High(torch.cat([x_LH,x_HL,x_HH],dim=1)).flatten(2).transpose(1,2)
        KV_H = self.KV_High(x_H)
        KV_H = KV_H.reshape(B_, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        KH, VH = KV_H[0], KV_H[1]

        # cosine attention
        A_L = (F.normalize(Q, dim=-1) @ F.normalize(KL, dim=-1).transpose(-2, -1))
        logit_scale_Low = torch.clamp(self.logit_scale_Low, max=torch.log(torch.tensor(1. / 0.01))).exp()
        A_L = A_L * logit_scale_Low
        A_L = self.softmax(A_L)
        A_L = self.attn_drop_L(A_L)
        A_L = (A_L @ VL).permute(0,2,3,1).reshape(-1, N, C)

        A_H = (F.normalize(Q, dim=-1) @ F.normalize(KH, dim=-1).transpose(-2, -1))
        logit_scale_High = torch.clamp(self.logit_scale_High, max=torch.log(torch.tensor(1. / 0.01))).exp()
        A_H = A_H * logit_scale_High
        A_H = self.softmax(A_H)
        A_H = self.attn_drop_H(A_H)
        A_H = (A_H @ VH).permute(0,2,3,1).reshape(-1, N, C)

        # Convolution Supplementary Part (CSP)
        x_LL_conv = self.L_conv(window_reverse(x_LL.permute(0,2,3,1),self.window_size[0]//2,H//2,W//2).permute(0,3,1,2))
        x_LH_conv = self.H_conv(window_reverse(x_LH.permute(0,2,3,1),self.window_size[0]//2,H//2,W//2).permute(0,3,1,2))
        x_HL_conv = self.H_conv(window_reverse(x_HL.permute(0,2,3,1),self.window_size[0]//2,H//2,W//2).permute(0,3,1,2))
        x_HH_conv = self.H_conv(window_reverse(x_HH.permute(0,2,3,1),self.window_size[0]//2,H//2,W//2).permute(0,3,1,2))

        Rst = window_partition(self.Rst_conv(self.IDWT(self.merge_IDWT(x_LL_conv,x_LH_conv,x_HL_conv,x_HH_conv))).permute(0,2,3,1),self.window_size[0]).permute(0,3,1,2).flatten(2).transpose(1,2)
        A_LH = self.fuse_All(torch.cat([A_L, A_H, Rst], dim=-1))
        x = self.proj(A_LH)
        x = self.proj_drop(x)
        return x

    def split_DWT(self, input):
        low_freq  = input[0]
        high_freq = input[1]
        h1 = high_freq[0][:, :, 0, :, :]
        h2 = high_freq[0][:, :, 1, :, :]
        h3 = high_freq[0][:, :, 2, :, :]
        return low_freq,h1,h2,h3

    def merge_IDWT(self, l,h1,h2,h3):
        high_freqs = []
        high_freqs.append(h1.unsqueeze(dim=2))
        high_freqs.append(h2.unsqueeze(dim=2))
        high_freqs.append(h3.unsqueeze(dim=2))
        high_freqs = torch.cat(high_freqs,dim=2)

        return l, [high_freqs]


class Efficient_Channel_Self_Attention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (F.normalize(q, dim=-2).transpose(-2, -1) @ F.normalize(k, dim=-2))
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01))).exp()
        attn = attn * logit_scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class Efficient_Channel_ViT(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, cpe_act=False):
        super().__init__()
        self.cpe = nn.ModuleList([ConvPosEnc(dim=dim, k=3, act=cpe_act),
                                  ConvPosEnc(dim=dim, k=3, act=cpe_act)])
        self.norm1 = norm_layer(dim)
        self.ECSA = Efficient_Channel_Self_Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer)

    def forward(self, x, H,W):
        #B L C
        x = self.cpe[0](x, H,W)
        x = x + self.drop_path(self.norm1(self.ECSA(x)))
        x = self.cpe[1](x, H,W)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class Wave_Conv_ViT(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size, mlp_ratio=4., drop=0., attn_drop=0., drop_path=0.,act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.cpe = nn.ModuleList([ConvPosEnc(dim=dim, k=3),ConvPosEnc(dim=dim, k=3)])
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.window_size = min(self.input_resolution)

        self.norm1 = norm_layer(dim)
        self.WCSA = Wave_Conv_Self_Attention(dim=dim, window_size=to_2tuple(self.window_size), num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x,H,W):
        shortcut=None
        if len(x.shape)==3:
            B,L,C = x.shape
            x = self.cpe[0](x,H,W)
            shortcut = x
            x = x.view(B, H, W, C).contiguous()
        elif len(x.shape) == 4:
            B,H,W,C = x.shape
            x = x.view(B, H*W, C).contiguous()
            x = self.cpe[0](x, H, W)
            shortcut = x
            x = x.view(B,H,W,C).contiguous()

        # partition windows
        x_windows = window_partition(x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # WCSA
        attn_windows = self.WCSA(x_windows,H,W)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(self.norm1(x))
        x = self.cpe[1](x,H,W)

        # FFN
        x = x + self.drop_path(self.norm2(self.mlp(x)))

        return x

class Block(nn.Module):
    def __init__(self, dim,num_blocks, input_resolution, num_heads, window_size,drop_path):
        super().__init__()
        self.WCV_blocks = nn.ModuleList([Wave_Conv_ViT(dim=dim,input_resolution=input_resolution,
                                  num_heads=num_heads,  window_size=window_size, drop_path=drop_path[i])
            for i in range(num_blocks)])
        self.ECV = Efficient_Channel_ViT(dim=dim,num_heads=num_heads)

    def forward(self, x):
        B,C,H,W = x.shape
        x = x.permute(0, 2, 3, 1)
        for WCV in self.WCV_blocks:
            x= WCV(x, H, W)
        x = self.ECV(x,H,W)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        return x

class Bidirectional_Interactive_Multi_Scale_Network(nn.Module):
    def __init__(self, dim,num_enc_layers = [2, 2, 2, 2, 2],num_heads=2,input_resolution=128,window_size=8):
        super().__init__()
        self.convert = nn.Sequential(nn.Conv2d(3, dim, 3, 1, 1), nn.LeakyReLU(0.2))
        self.convert_s = nn.Sequential(nn.Conv2d(3, dim, 3, 1, 1), nn.LeakyReLU(0.2))
        self.wavelet_fusion1 = convd(dim * 4, dim, 3, 1)
        self.wavelet_fusion2 = convd(dim * 4, dim, 3, 1)
        self.wavelet_fusion3 = convd(dim * 4, dim, 3, 1)
        self.wavelet_fusion4 = convd(dim * 4, dim, 3, 1)
        self.wavelet_expand1 = convd(dim , dim * 4, 3, 1)
        self.wavelet_expand2 = convd(dim , dim * 4, 3, 1)
        self.wavelet_expand3 = convd(dim , dim * 4, 3, 1)
        self.wavelet_expand4 = convd(dim , dim * 4, 3, 1)
        self.skip_fusion0 = convd(dim * 2, dim, 3, 1)
        self.skip_fusion1 = convd(dim * 2, dim, 3, 1)
        self.skip_fusion2 = convd(dim * 2, dim, 3, 1)
        self.skip_fusion3 = convd(dim * 2, dim, 3, 1)
        self.out = nn.Sequential(nn.Conv2d(dim, dim, 3, 1, 1), nn.LeakyReLU(0.2),
                                 nn.Conv2d(dim, 3, 1, 1))
        self.drop_path_rate = 0.1

        total_blocks = sum(num_enc_layers)  # 10
        self.enc_dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, total_blocks)]

        self.dec_dpr = self.enc_dpr[::-1]
        self.Encoder0 = Block(
            dim=dim,
            num_blocks=num_enc_layers[0],
            input_resolution=to_2tuple(input_resolution),
            num_heads=num_heads,
            window_size=window_size,
            drop_path=self.enc_dpr[0: num_enc_layers[0]]
        )

        self.Encoder1 = Block(
            dim=dim,
            num_blocks=num_enc_layers[1],
            input_resolution=to_2tuple(input_resolution // 2),
            num_heads=num_heads,
            window_size=window_size,
            drop_path=self.enc_dpr[
                      num_enc_layers[0]: num_enc_layers[0] + num_enc_layers[1]
                      ]
        )

        self.Encoder2 = Block(
            dim=dim,
            num_blocks=num_enc_layers[2],
            input_resolution=to_2tuple(input_resolution // 4),
            num_heads=num_heads,
            window_size=window_size,
            drop_path=self.enc_dpr[
                      num_enc_layers[0] + num_enc_layers[1]:
                      num_enc_layers[0] + num_enc_layers[1] + num_enc_layers[2]
                      ]
        )

        self.Encoder3 = Block(
            dim=dim,
            num_blocks=num_enc_layers[3],
            input_resolution=to_2tuple(input_resolution // 8),
            num_heads=num_heads,
            window_size=window_size,
            drop_path=self.enc_dpr[
                      num_enc_layers[0] + num_enc_layers[1] + num_enc_layers[2]:
                      num_enc_layers[0] + num_enc_layers[1] + num_enc_layers[2] + num_enc_layers[3]
                      ]
        )

        self.Encoder4 = Block(
            dim=dim,
            num_blocks=num_enc_layers[4],
            input_resolution=to_2tuple(input_resolution // 16),
            num_heads=num_heads,
            window_size=window_size,
            drop_path=self.enc_dpr[
                      num_enc_layers[0] + num_enc_layers[1] + num_enc_layers[2] + num_enc_layers[3]:
                      num_enc_layers[0] + num_enc_layers[1] + num_enc_layers[2] + num_enc_layers[3] + num_enc_layers[4]
                      ]
        )

        self.Decoder4 = Block(
            dim=dim,
            num_blocks=num_enc_layers[4],
            input_resolution=to_2tuple(input_resolution // 16),
            num_heads=num_heads,
            window_size=window_size,
            drop_path=self.dec_dpr[0: num_enc_layers[4]]
        )

        self.Decoder3 = Block(
            dim=dim,
            num_blocks=num_enc_layers[3],
            input_resolution=to_2tuple(input_resolution // 8),
            num_heads=num_heads,
            window_size=window_size,
            drop_path=self.dec_dpr[
                      num_enc_layers[4]: num_enc_layers[4] + num_enc_layers[3]
                      ]
        )

        self.Decoder2 = Block(
            dim=dim,
            num_blocks=num_enc_layers[2],
            input_resolution=to_2tuple(input_resolution // 4),
            num_heads=num_heads,
            window_size=window_size,
            drop_path=self.dec_dpr[
                      num_enc_layers[4] + num_enc_layers[3]:
                      num_enc_layers[4] + num_enc_layers[3] + num_enc_layers[2]
                      ]
        )

        self.Decoder1 = Block(
            dim=dim,
            num_blocks=num_enc_layers[1],
            input_resolution=to_2tuple(input_resolution // 2),
            num_heads=num_heads,
            window_size=window_size,
            drop_path=self.dec_dpr[
                      num_enc_layers[4] + num_enc_layers[3] + num_enc_layers[2]:
                      num_enc_layers[4] + num_enc_layers[3] + num_enc_layers[2] + num_enc_layers[1]
                      ]
        )

        self.Decoder0 = Block(
            dim=dim,
            num_blocks=num_enc_layers[0],
            input_resolution=to_2tuple(input_resolution),
            num_heads=num_heads,
            window_size=window_size,
            drop_path=self.dec_dpr[
                      num_enc_layers[4] + num_enc_layers[3] + num_enc_layers[2] + num_enc_layers[1]:
                      total_blocks
                      ]
        )
        self.DWT = DWTForward(J=1, wave='haar').cuda()
        self.IDWT = DWTInverse(wave='haar').cuda()
        self.SAFB0 = Spatial_Adaptive_Fusion_Block(dim)
        self.SAFB1 = Spatial_Adaptive_Fusion_Block(dim)
        self.SAFB2 = Spatial_Adaptive_Fusion_Block(dim)
        self.SAFB3 = Spatial_Adaptive_Fusion_Block(dim)
        self.SAFB4 = Spatial_Adaptive_Fusion_Block(dim)


    def forward(self, x):
        x_check, ori_size = self.check_image_size(x)
        EI_0 = self.convert(x_check)

        EO_0 = self.Encoder0(EI_0)
        EI_1 = self.wavelet_fusion1(self.split_DWT_cat(self.DWT(EO_0)))

        EO_1 = self.Encoder1(EI_1)
        EI_2 = self.wavelet_fusion2(self.split_DWT_cat(self.DWT(EO_1)))

        EO_2 = self.Encoder2(EI_2)
        EI_3 = self.wavelet_fusion3(self.split_DWT_cat(self.DWT(EO_2)))

        EO_3 = self.Encoder3(EI_3)
        EI_4 = self.wavelet_fusion4(self.split_DWT_cat(self.DWT(EO_3)))

        EO_4 = self.Encoder4(EI_4)

        _,_,h4,w4 = EO_4.shape
        JI_4 = self.convert_s(F.interpolate(x_check, size=[h4, w4], mode='bilinear'))
        JI_3, JO_4 = self.SAFB4(JI_4, EO_4)

        DO_4 = self.Decoder4(JO_4)
        DI_3= self.IDWT(self.merge_IDWT_cat(self.wavelet_expand4(DO_4)))
        JI_2, JO_3 = self.SAFB3(JI_3, EO_3)
        DI_3 = self.skip_fusion3(torch.cat([DI_3,JO_3],dim=1))
        DO_3 = self.Decoder3(DI_3)

        DI_2= self.IDWT(self.merge_IDWT_cat(self.wavelet_expand3(DO_3)))
        JI_1, JO_2 = self.SAFB2(JI_2, EO_2)
        DI_2 = self.skip_fusion2(torch.cat([DI_2,JO_2],dim=1))
        DO_2 = self.Decoder2(DI_2)

        DI_1= self.IDWT(self.merge_IDWT_cat(self.wavelet_expand2(DO_2)))
        JI_0, JO_1 = self.SAFB1(JI_1, EO_1)
        DI_1 = self.skip_fusion1(torch.cat([DI_1,JO_1],dim=1))
        DO_1 = self.Decoder1(DI_1)

        DI_0= self.IDWT(self.merge_IDWT_cat(self.wavelet_expand1(DO_1)))
        _, JO_0 = self.SAFB0(JI_0, EO_0)
        DI_0 = self.skip_fusion0(torch.cat([DI_0,JO_0],dim=1))

        DO_0 = self.Decoder0(DI_0)

        y = self.out(DO_0)
        out = x_check + y
        out = self.restore_image_size(out, ori_size)

        return out

    def check_image_size(self, x):
        _, _, h, w = x.size()
        size = 128
        mod_pad_h = (size - h % size) % size
        mod_pad_w = (size - w % size) % size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        ori_size = [h, w]
        return x, ori_size

    def restore_image_size(self, x, ori_size):
        return x[:, :, :ori_size[0], :ori_size[1]]

    def split_DWT_cat(self, x):
        low_freq = x[0]
        high_freq = x[1]
        h1 = high_freq[0][:, :, 0, :, :]
        h2 = high_freq[0][:, :, 1, :, :]
        h3 = high_freq[0][:, :, 2, :, :]
        out = torch.cat([low_freq,h1,h2,h3],dim=1)
        return out

    def merge_IDWT_cat(self, x):
        channel = x.shape[1] // 4
        l = x[:, 0:channel, :, :]
        h1 = x[:, channel:channel * 2, :, :]
        h2 = x[:, channel * 2:channel * 3, :, :]
        h3 = x[:, channel * 3:channel * 4, :, :]
        high_freqs = []
        high_freqs.append(h1.unsqueeze(dim=2))
        high_freqs.append(h2.unsqueeze(dim=2))
        high_freqs.append(h3.unsqueeze(dim=2))
        high_freqs  = torch.cat(high_freqs,dim=2)

        return l, [high_freqs]



