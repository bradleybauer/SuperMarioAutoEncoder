import torch as th
import torch.nn as nn
import torchvision as tv
from torchinfo import summary
from torch import Tensor

from typing import Callable, List, Optional

def ms(m, ins=(1,3,180,180)):
    return summary(m, input_size=ins, col_names = [
            "input_size",
            "output_size",
            "num_params",
            "kernel_size",
        ])


def mix(a,b,m):
    return a + (b-a) * m

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1, output_padding: int = 0) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.ConvTranspose2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        output_padding = output_padding,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def conv1x1(in_planes: int, out_planes: int, stride: int = 1, output_padding: int = 0) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, output_padding = output_padding)

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 2

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        output_padding: int = 0,
        upsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.upsample layers upsample the input when stride != 1
        self.conv3 = conv1x1(planes * self.expansion, width)
        self.bn3 = norm_layer(planes)

        self.conv2 = conv3x3(width, width, stride, groups, dilation, output_padding)
        self.bn2 = norm_layer(width)

        self.conv1 = conv1x1(width, inplanes)
        self.bn1 = norm_layer(inplanes)
        
        
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv3(x)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn1(out)
        if self.upsample is not None:
            identity = self.upsample(x)
        out += identity
        out = self.relu(out)

        return out

## resnet code modified to generate images. taken from somewhere on the internet.
## tweaked to work with resolutions specific to my AutoEncoder
class ResNet(nn.Module):
    def __init__(
        self,
        BlockType,
        layers: List[int],
        sizes: List[int],
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        # indices = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.de_conv1 = nn.ConvTranspose2d(64, 3, kernel_size=7, stride=1, padding=3, bias=False)
        self.relu = nn.ReLU(inplace=True)

        # sizes =   [180,90,45,20,8,4][::-1]
        self.up1 = nn.Upsample(size=sizes[0], mode='nearest') # 4
        self.layer4 = self._make_layer(BlockType, 576, layers[3], stride=2) # 8
        self.up2 = nn.Upsample(size=sizes[2], mode='nearest') # 20
        self.layer3 = self._make_layer(BlockType, 288, layers[2], output_padding=0)
        self.up3 = nn.Upsample(size=sizes[3], mode='nearest') # 45
        self.layer2 = self._make_layer(BlockType, 144, layers[1], stride=2) # 90
        self.layer1 = self._make_layer(BlockType, 72, layers[0], stride=2, last_block_dim=64) # 180

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        BlockType,
        planes: int,
        blocks: int,
        stride: int = 1,
        output_padding: int = 1,
        last_block_dim: int = 0,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        upsample = None
        previous_dilation = self.dilation

        layers = []
        self.inplanes = planes * BlockType.expansion
        if last_block_dim == 0:
            last_block_dim = self.inplanes//2
        if stride != 1 or self.inplanes != planes * BlockType.expansion or output_padding==0:
            upsample = nn.Sequential(
                conv1x1(planes * BlockType.expansion, last_block_dim, stride, output_padding),
                # norm_layer(planes * BlockType.expansion),
                norm_layer(last_block_dim),
            )
        last_block = BlockType(last_block_dim, planes, stride, output_padding, upsample, self.groups, self.base_width, previous_dilation, norm_layer)

        for _ in range(1, blocks):
            layers.append(
                BlockType(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )
        layers.append(last_block)
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.up1(x)
        x = self.layer4(x)
        x = self.up2(x)
        x = self.layer3(x)
        x = self.up3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = self.de_conv1(x)
        x = self.relu(x)
        return x

## my code follows

class VAE(nn.Module):
    def __init__(self, in_sz):
        super().__init__()
        # self.to_mean = nn.Linear(in_dim,latent_dim)
        # self.to_logvar = nn.Linear(in_dim,latent_dim)
        C,H,W = in_sz
        self.to_mean = nn.Sequential(
            nn.Conv2d(C,C,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(C,C,kernel_size=3,stride=1,padding=1),
        )
        self.to_logvar = nn.Sequential(
            nn.Conv2d(C,C,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(C,C,kernel_size=3,stride=1,padding=1),
        )
        # self.do = nn.Dropout2d(.25)
        self.ema_mean = nn.Parameter(th.zeros(C*H*W), requires_grad=False)
        self.ema_logvar = nn.Parameter(th.zeros(C*H*W), requires_grad=False)
        self.latent_dim = C*H*W

    def forward(self, x, s):
        N,C,H,W = x.shape
        # x = self.do(x)
        mean, _ = self.to_mean((x,s))
        logvar, _ = self.to_logvar((x,s))
        mean = mean.view(N,-1)
        logvar = logvar.view(N,-1)
        if self.training:
            self.ema_mean.data = mix(self.ema_mean.data, mean.detach().mean(dim=0), .0001)
            self.ema_logvar.data = mix(self.ema_logvar.data, logvar.detach().mean(dim=0), .0001)
        z = th.randn_like(mean) * th.exp(.5 * logvar) + mean
        return z, mean, logvar

    def sample(self, N):
        eps = th.randn(N, self.latent_dim).cuda()
        return eps * th.exp(.5 * self.ema_logvar.data) + self.ema_mean.data

class ConditionedLinear(nn.Module):
    def __init__(self, channels, state_dim=37, renorm=False):
        super().__init__()
        self.lin = nn.Linear(state_dim, channels, bias=False)
        if renorm:
            self.lin2 = nn.Linear(state_dim, channels, bias=False)
        self.renorm = renorm
        # self.do = nn.Dropout2d()

    def set_state(self, s):
        self.s = s

    def forward(self, x):
        n,c,_,_ = x.shape
        if self.renorm:
            eps = .00001
            new_mean = self.lin(self.s).view(n,c,1,1)
            new_std = (self.lin2(self.s).view(n,c,1,1)**2 + eps).sqrt()
            mean = x.mean(dim=(2,3)).view(n,c,1,1)
            std = (x.var(dim=(2,3)) + eps).sqrt().view(n,c,1,1)
            return new_std * ((x - mean) / std) + new_mean
        else:
            return x + self.lin(self.s).view(n,c,1,1)

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = tv.models.resnet18(weights=None)
        self.encoder.avgpool = nn.AdaptiveAvgPool2d(output_size=(3,3)) # final activation has 4608 numbers in it
        self.encoder.fc = nn.Identity()
        self.encoder.layer1 = nn.Sequential(self.encoder.layer1, ConditionedLinear(64))
        self.encoder.layer2 = nn.Sequential(self.encoder.layer2, ConditionedLinear(128))
        self.encoder.layer3 = nn.Sequential(self.encoder.layer3, ConditionedLinear(256))
        self.encoder.layer4 = nn.Sequential(self.encoder.layer4, ConditionedLinear(512))

        sizes = [180,90,45,20,8,4]
        self.decoder = ResNet(Bottleneck, [2, 2, 2, 2], sizes[::-1])
        self.decoder.layer1 = nn.Sequential(self.decoder.layer1, ConditionedLinear(64))
        self.decoder.layer2 = nn.Sequential(self.decoder.layer2, ConditionedLinear(144))
        self.decoder.layer3 = nn.Sequential(self.decoder.layer3, ConditionedLinear(288))
        self.decoder.layer4 = nn.Sequential(self.decoder.layer4, ConditionedLinear(576))
        # print(self.decoder)
        # print(ms(self.decoder, ins=(1,1152,2,2)))
        # exit()

        self.encoder_condition = [self.encoder.layer1[1], self.encoder.layer2[1], self.encoder.layer3[1], self.encoder.layer4[1]]
        self.decoder_condition = [self.decoder.layer1[1], self.decoder.layer2[1], self.decoder.layer3[1], self.decoder.layer4[1]]

        self._encode_only = False

    def forward(self, x, s):
        N,C,H,W = x.shape

        for linear in self.encoder_condition + self.decoder_condition:
            linear.set_state(s)

        x = self.encoder(x).view(N,1152,2,2)
        x = self.decoder(x)
        return {
                'reconstruction':x,
            }
    
    def encode(self, x, s):
        N = x.shape[0]
        for linear in self.encoder_condition:
            linear.set_state(s)
        return self.encoder(x).view(N,-1)
    
    def decode(self, x, s):
        N = x.shape[0]
        for linear in self.decoder_condition:
            linear.set_state(s)
        return self.decoder(x.view(N,1152,2,2))

    def sample(self, N):
        z = self.vae.sample(N)
        z = z.view(N,512,3,3)
        x = self.decoder(z)
        return x

    def grad_off(self):
        for p in self.parameters():
            p.requires_grad = False
    def grad_on(self):
        for p in self.parameters():
            p.requires_grad = True
