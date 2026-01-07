from torch import nn
import torch
from torchvision import models


def conv3x3(in_ch, out_ch, s=1, d=1):
    p = (3 // 2) * d
    return nn.Conv2d(in_ch, out_ch, 3, stride=s, padding=p, dilation=d, bias=True)

class ConvINLReLU(nn.Module):
    def __init__(self, in_ch, out_ch, s=1, d=1, negative_slope=0.1):
        super().__init__()
        self.conv = conv3x3(in_ch, out_ch, s=s, d=d)
        self.inorm = nn.InstanceNorm2d(out_ch, affine=True)
        self.act = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

    def forward(self, x):
        return self.act(self.inorm(self.conv(x)))

class MRM(nn.Module):
    def __init__(self, ch):
        super().__init__()
        c1 = ch // 2
        c2 = ch
        c3 = int(1.5 * ch)
        self.b1 = ConvINLReLU(ch, c1, d=1)
        self.b2 = ConvINLReLU(c1, c2, d=1)
        self.b3 = ConvINLReLU(c2, c3, d=1)
        self.merge = ConvINLReLU(c1 + c2 + c3, ch, d=1)
        self.skip = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        f1 = self.b1(x)
        f2 = self.b2(f1)
        f3 = self.b3(f2)
        return self.merge(torch.cat([f1, f2, f3], 1)) + self.skip(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=True),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(0.1, True),
            MRM(out_ch)
        )

    def forward(self, x):
        return self.conv(self.pool(x))

class TCUM(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)

    def forward(self, x):
        return self.up(x)

class SOSFM(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.refine = MRM(out_ch)
        self.align = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, enc_feat_same_level, up_feat):
        i_n = self.align(enc_feat_same_level)
        fused = i_n + up_feat
        return self.refine(fused) - up_feat

class DualDomainFusion(nn.Module):
    """
    Dual-domain fusion at bottleneck:
    spatial path + frequency path with channel-wise magnitude modulation.
    FFT and polar are forced to float32 to avoid AMP half-precision issues.
    """
    def __init__(self, ch, r=4):
        super().__init__()
        self.spatial = nn.Sequential(
            ConvINLReLU(ch, ch),
            ConvINLReLU(ch, ch)
        )
        mid = max(8, ch // r)
        self.freq_ca = nn.Sequential(
            nn.Conv2d(ch, mid, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid, ch, 1),
            nn.Sigmoid()
        )
        self.fuse = ConvINLReLU(2 * ch, ch)
        self.res = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        # Save original dtype (e.g. float16 under AMP)
        orig_dtype = x.dtype

        # Work in float32 inside the module
        x32 = x.to(torch.float32)

        # Spatial path (float32)
        x_sp = self.spatial(x32)

        # Frequency path (float32)
        X = torch.fft.fft2(x32, dim=(-2, -1))   # complex64
        mag = torch.abs(X)
        pha = torch.angle(X)

        w = self.freq_ca(mag.mean(dim=(-2, -1), keepdim=True))
        mag2 = mag * w

        X2 = torch.polar(mag2, pha)            # complex64, float-safe
        x_fr = torch.fft.ifft2(X2, dim=(-2, -1)).real  # float32

        out32 = self.fuse(torch.cat([x_sp, x_fr], dim=1))
        out32 = self.res(out32)

        # Cast back to original dtype for the rest of the network
        out = out32.to(orig_dtype)
        return out + x

#=====================================
# DDSF-UNet Model
#=====================================  
class DDSF_UNet(nn.Module):
    
    def __init__(self, base=32, in_ch=3, out_ch=3):
        super().__init__()
        c1, c2, c3, c4, c5 = base, base * 2, base * 4, base * 8, base * 8

        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, c1, 3, padding=1, bias=True),
            nn.InstanceNorm2d(c1, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
            MRM(c1)
        )
        self.down1 = Down(c1, c2)
        self.down2 = Down(c2, c3)
        self.down3 = Down(c3, c4)

        self.bot_mrm = MRM(c4)
        self.bot_ddf = DualDomainFusion(c4)
        #self.bot_ddf = nn.Identity()
        self.bot_out = nn.Conv2d(c4, c5, 3, padding=1, bias=True)

        self.up3 = TCUM(c5, c4)
        self.sos3 = SOSFM(in_ch=c3, out_ch=c4)

        self.up2 = TCUM(c4, c3)
        self.sos2 = SOSFM(in_ch=c2, out_ch=c3)

        self.up1 = TCUM(c3, c2)
        self.sos1 = SOSFM(in_ch=c1, out_ch=c2)

        self.head = nn.Conv2d(c2, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.down1(e1)
        e3 = self.down2(e2)
        e4 = self.down3(e3)

        b = self.bot_out(self.bot_ddf(self.bot_mrm(e4)))

        d3 = self.sos3(e3, self.up3(b))
        d2 = self.sos2(e2, self.up2(d3))
        d1 = self.sos1(e1, self.up1(d2))

        y = self.head(d1)
        return torch.tanh(y) * 0.5 + 0.5


#=====================================
# Perceptual VGG19 feature extractor
#=====================================

class PerceptualVGG19(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            weights = models.VGG19_Weights.IMAGENET1K_FEATURES
            vgg = models.vgg19(weights=weights).features
        except Exception:
            vgg = models.vgg19(pretrained=True).features
        for p in vgg.parameters():
            p.requires_grad = False
        self.feat = nn.Sequential(*[vgg[i] for i in range(36)])
        for p in self.feat.parameters():
            p.requires_grad = False

    def forward(self, x):
        mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
        std = x.new_tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)
        return self.feat((x - mean) / std)
