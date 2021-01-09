
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1,
                 dilation=1, groups=1, bias=False):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
                in_chan, out_chan, kernel_size=ks, stride=stride,
                padding=padding, dilation=dilation,
                groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv(x)
        feat = self.bn(feat)
        feat = self.relu(feat)
        return feat

class DSConv(nn.Module):
    """Depthwise Separable convolutions"""
    def __init__(self, dw_channels, out_channels, stride=1, dilation=1, padding=1):
        super(DSConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, dw_channels, 3, stride, padding=padding, groups=dw_channels, dilation=dilation, bias=False),
            nn.BatchNorm2d(dw_channels),
            nn.ReLU(True),
            nn.Conv2d(dw_channels,out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
    
    def forward(self, x):
        return self.conv(x)

class StemBlock_Shared(nn.Module):

    def __init__(self):
        super(StemBlock_Shared, self).__init__()
        self.conv = ConvBNReLU(3, 16, 3, stride=1, dilation=1)
        self.left = nn.Sequential(
            ConvBNReLU(16, 8, 1, stride=1, padding=0),
            DSConv(8, 16, stride=2),
        )
        self.right = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, ceil_mode=False)
        self.fuse = DSConv(32, 16, stride=1)

    def forward(self, x):
        feat = self.conv(x)
        feat_left = self.left(feat)
        feat_right = self.right(feat)
        feat = torch.cat([feat_left, feat_right], dim=1)
        feat = self.fuse(feat)
        return feat # H/2, W/2, 16

class LearningToDownsample(nn.Module):
    """Learning to downsample module """

    def __init__(self, dw_channels1=32, dw_channels2=48, out_channels=64):
        super(LearningToDownsample, self).__init__()
        self.stem = StemBlock_Shared() # this is replaced by the shared LTDS, but should can expect to get H/2, H/2, 16
        # self.conv = ConvBNReLU(3, dw_channels1,3,2) # /2
        
        padding_1 = ((dw_channels1-1)*1 - 16 + 2*(3-1)+1)/2.0
        padding_2 = ((128-1)*1 - 128 + 5)/2.0

        self.dsconv1 = DSConv(16, dw_channels1) # /2, 32        
        self.dsconv2 = DSConv(dw_channels1, dw_channels2, stride=2, dilation=1) # /4, 48
        self.dsconv3= DSConv(dw_channels2, 16, 1, 1) # /4, 16

        self.ge2 = GELayerS2(dw_channels2, out_channels) # /8, 64
        self.dsconv4 = DSConv(out_channels, 128, 1) # /8, 128
        self.dsconv5 = DSConv(128, 128, 1, 2, int(padding_2)) #/8, 128
        self.dsconv6 = DSConv(128, 128)

    def forward(self, x):
        shared = self.stem(x) # /2, 16
        shared = self.dsconv1(shared) # /2, 32
        shared = self.dsconv2(shared) # /4, 48
        semantic = self.dsconv3(shared) # /4, 16
        detail = self.ge2(shared) # /8, 64
        detail = self.dsconv4(detail) # /8, 128
        detail = self.dsconv5(detail) # /8, 128, dilated
        detail = self.dsconv6(detail)
        return semantic, detail

class Classifier(nn.Module):
    """Classifier module replaces SegmentHead"""

    def __init__(self, dw_channels, mid_channels, num_classes, up_factor=8, aux=True):
        super(Classifier, self).__init__()
        out_chan = num_classes * up_factor * up_factor
        self.up_factor = up_factor
        self.dsconv1 = DSConv(dw_channels, mid_channels, 1)
        self.dsconv2 = DSConv(mid_channels, up_factor*up_factor, 1)
        self.conv_out = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(up_factor*up_factor, out_chan, 1),
            nn.PixelShuffle(up_factor)
        )
        
    
    def forward(self, x):
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        x = self.conv_out(x)
        return x


class UpSample(nn.Module):

    def __init__(self, n_chan, factor=2):
        super(UpSample, self).__init__()
        out_chan = n_chan * factor * factor
        self.proj = nn.Conv2d(n_chan, out_chan, 1, 1, 0)
        self.up = nn.PixelShuffle(factor)
        self.init_weight()

    def forward(self, x):
        feat = self.proj(x)
        feat = self.up(feat)
        return feat

    def init_weight(self):
        nn.init.xavier_normal_(self.proj.weight, gain=1.)


class DetailBranch(nn.Module):

    def __init__(self):
        super(DetailBranch, self).__init__()
        self.S1 = nn.Sequential(
            ConvBNReLU(3, 64, 3, stride=2),
            ConvBNReLU(64, 64, 3, stride=1),
        )
        self.S2 = nn.Sequential(
            ConvBNReLU(64, 64, 3, stride=2),
            ConvBNReLU(64, 64, 3, stride=1),
            ConvBNReLU(64, 64, 3, stride=1),
        )
        self.S3 = nn.Sequential(
            ConvBNReLU(64, 128, 3, stride=2),
            ConvBNReLU(128, 128, 3, stride=1),
            ConvBNReLU(128, 128, 3, stride=1),
        )

    def forward(self, x):
        feat = self.S1(x)
        feat = self.S2(feat)
        feat = self.S3(feat)
        return feat


class StemBlock(nn.Module):

    def __init__(self):
        super(StemBlock, self).__init__()
        self.conv = ConvBNReLU(3, 16, 3, stride=2)
        self.left = nn.Sequential(
            ConvBNReLU(16, 8, 1, stride=1, padding=0),
            ConvBNReLU(8, 16, 3, stride=2),
        )
        self.right = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, ceil_mode=False)
        self.fuse = ConvBNReLU(32, 16, 3, stride=1)

    def forward(self, x):
        feat = self.conv(x)
        feat_left = self.left(feat)
        feat_right = self.right(feat)
        feat = torch.cat([feat_left, feat_right], dim=1)
        feat = self.fuse(feat)
        return feat # H/4/ W/4, 16


class CEBlock(nn.Module):

    def __init__(self):
        super(CEBlock, self).__init__()
        self.bn = nn.BatchNorm2d(128)
        self.conv_gap = ConvBNReLU(128, 128, 1, stride=1, padding=0)
        #TODO: in paper here is naive conv2d, no bn-relu
        # self.conv_last = ConvBNReLU(128, 128, 3, stride=1)
        self.conv_last = nn.Conv2d(
                128, 128, kernel_size=3, stride=1,
                padding=1, groups=128, bias=False)
       

    def forward(self, x):
        feat = torch.mean(x, dim=(2, 3), keepdim=True)
        feat = self.bn(feat)
        feat = self.conv_gap(feat)
        feat = feat + x
        feat = self.conv_last(feat)
        return feat


class GELayerS1(nn.Module):

    def __init__(self, in_chan, out_chan, exp_ratio=6, dilation=1, padding=1):
        super(GELayerS1, self).__init__()
        mid_chan = in_chan * exp_ratio
        self.conv1 = ConvBNReLU(in_chan, in_chan, 3, stride=1)
        self.dwconv = nn.Sequential(
            nn.Conv2d(
                in_chan, mid_chan, kernel_size=3, stride=1,
                padding=padding, dilation=dilation, groups=in_chan, bias=False),
            nn.BatchNorm2d(mid_chan),
            # nn.ReLU(inplace=True), # not shown in paper
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                mid_chan, out_chan, kernel_size=1, stride=1,
                padding=0, bias=False),
            nn.BatchNorm2d(out_chan),
        )
        self.conv2[1].last_bn = True
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.dwconv(feat)
        feat = self.conv2(feat)
        feat = feat + x
        feat = self.relu(feat)
        return feat


class GELayerS2(nn.Module):

    def __init__(self, in_chan, out_chan, exp_ratio=6):
        super(GELayerS2, self).__init__()
        mid_chan = in_chan * exp_ratio
        self.conv1 = ConvBNReLU(in_chan, in_chan, 3, stride=1)
        self.dwconv1 = nn.Sequential(
            nn.Conv2d(
                in_chan, mid_chan, kernel_size=3, stride=2,
                padding=1, groups=in_chan, bias=False),
            nn.BatchNorm2d(mid_chan),
        )
        self.dwconv2 = nn.Sequential(
            nn.Conv2d(
                mid_chan, mid_chan, kernel_size=3, stride=1,
                padding=1, groups=mid_chan, bias=False),
            nn.BatchNorm2d(mid_chan),
            # nn.ReLU(inplace=True), # not shown in paper
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                mid_chan, out_chan, kernel_size=1, stride=1,
                padding=0, bias=False),
            nn.BatchNorm2d(out_chan),
        )
        self.conv2[1].last_bn = True
        self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_chan, in_chan, kernel_size=3, stride=2,
                    padding=1, groups=in_chan, bias=False),
                nn.BatchNorm2d(in_chan),
                nn.Conv2d(
                    in_chan, out_chan, kernel_size=1, stride=1,
                    padding=0, bias=False),
                nn.BatchNorm2d(out_chan),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.dwconv1(feat)
        feat = self.dwconv2(feat)
        feat = self.conv2(feat)
        shortcut = self.shortcut(x)
        feat = feat + shortcut
        feat = self.relu(feat)
        return feat


class SegmentBranch(nn.Module):

    def __init__(self):
        super(SegmentBranch, self).__init__()
        # self.S1S2 = StemBlock() # this is replaced by the shared LTDS
        self.S3 = nn.Sequential(
            GELayerS2(16, 32),
            GELayerS1(32, 32, 6, 2, 2), # dilated
        )
        self.S4 = nn.Sequential(
            GELayerS2(32, 64),
            GELayerS1(64, 64, 6, 2, 2), # dilated
        )
        self.S5_4 = nn.Sequential(
            GELayerS2(64, 128),
            GELayerS1(128, 128, 6, 2, 2), # dilated
            GELayerS1(128, 128),
            GELayerS1(128, 128),
        )
        self.S5_5 = CEBlock()

    def forward(self, x):
        # feat2 = self.S1S2(x)
        feat3 = self.S3(x) # x is giong to come in from the LTDS block now, as a H/8, W/8, 16 channel bock
        feat4 = self.S4(feat3)
        feat5_4 = self.S5_4(feat4)
        feat5_5 = self.S5_5(feat5_4)
        return feat3, feat4, feat5_4, feat5_5


class BGALayer(nn.Module):

    def __init__(self):
        super(BGALayer, self).__init__()
        self.left1 = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1,
                padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(
                128, 128, kernel_size=1, stride=1,
                padding=0, bias=False),
        )
        self.left2 = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=2,
                padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        )
        self.right1 = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1,
                padding=1, bias=False),
            nn.BatchNorm2d(128),
        )
        self.right2 = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1,
                padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(
                128, 128, kernel_size=1, stride=1,
                padding=0, bias=False),
        )
        self.up1 = nn.Upsample(scale_factor=4)
        self.up2 = nn.Upsample(scale_factor=4)
        ##TODO: does this really has no relu?
        self.conv = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1,
                padding=1, bias=False),
            nn.BatchNorm2d(128),
            # nn.ReLU(inplace=True), # not shown in paper
        )

    def forward(self, x_d, x_s):
        dsize = x_d.size()[2:]
        left1 = self.left1(x_d)
        left2 = self.left2(x_d)
        right1 = self.right1(x_s)
        right2 = self.right2(x_s)
        right1 = self.up1(right1)
        left = left1 * torch.sigmoid(right1)
        right = left2 * torch.sigmoid(right2)
        right = self.up2(right)
        out = self.conv(left + right)
        return out



class SegmentHead(nn.Module):

    def __init__(self, in_chan, mid_chan, n_classes, up_factor=8, aux=True):
        super(SegmentHead, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, 3, stride=1)
        self.drop = nn.Dropout(0.1)
        self.up_factor = up_factor

        out_chan = n_classes * up_factor * up_factor
        if aux:
            self.conv_out = nn.Sequential(
                ConvBNReLU(mid_chan, up_factor * up_factor, 3, stride=1),
                nn.Conv2d(up_factor * up_factor, out_chan, 1, 1, 0),
                nn.PixelShuffle(up_factor)
            )
        else:
            self.conv_out = nn.Sequential(
                nn.Conv2d(mid_chan, out_chan, 1, 1, 0),
                nn.PixelShuffle(up_factor)
            )

    def forward(self, x):
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        return feat


class BiSeNetV2(nn.Module):

    def __init__(self, n_classes, output_aux=True):
        super(BiSeNetV2, self).__init__()
        self.output_aux = output_aux
        self.ltds = LearningToDownsample(32, 48, 64) # returns semantic, detail branches
        
        # self.detail = DetailBranch() # this is replaced by the LTDS entirely
        self.segment = SegmentBranch()
        self.bga = BGALayer()

        ## TODO: what is the number of mid chan ?
        self.head = Classifier(128, 128, n_classes, up_factor=8, aux=False)
        # self.head = SegmentHead(128, 1024, n_classes, up_factor=8, aux=False)
        if self.output_aux:
            self.aux2 = SegmentHead(16, 128, n_classes, up_factor=4)
            self.aux3 = SegmentHead(32, 128, n_classes, up_factor=8)
            self.aux4 = SegmentHead(64, 128, n_classes, up_factor=16)
            self.aux5_4 = SegmentHead(128, 128, n_classes, up_factor=32)
    
        self.init_weights()

    def forward(self, x):
        size = x.size()[2:]
        # feat_d = self.detail(x)
        feat2, feat_d = self.ltds(x) # H/8, W/8, 64
        feat3, feat4, feat5_4, feat_s = self.segment(feat2)
        feat_head = self.bga(feat_d, feat_s)

        logits = self.head(feat_head)
        if self.output_aux:
            logits_aux2 = self.aux2(feat2)
            logits_aux3 = self.aux3(feat3)
            logits_aux4 = self.aux4(feat4)
            logits_aux5_4 = self.aux5_4(feat5_4)
            return logits, logits_aux2, logits_aux3, logits_aux4, logits_aux5_4
        pred = logits.argmax(dim=1)
        return pred

    def init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
                if not module.bias is None: nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                if hasattr(module, 'last_bn') and module.last_bn:
                    nn.init.zeros_(module.weight)
                else:
                    nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)


if __name__ == "__main__":
    x = torch.randn(16, 3, 1024, 2048)
    model = BiSeNetV2(n_classes=19)
    outs = model(x)
    for out in outs:
        print(out.size())
