import torch
import torch.nn as nn
import torch.nn.functional as F

class bn_lrelu(nn.Module):
    def __init__(self, input_ch, activation):
        super().__init__()
        self.bn = nn.BatchNorm2d(input_ch)
        if activation == "Relu":
            self.relu = nn.ReLU()
        elif activation == "Leaky_Relu":
            self.relu = nn.LeakyReLU(negative_slope=0.1)
        elif activation == "PRelu":
            self.relu = nn.PReLU()

    def forward(self, inputs):
        x = self.bn(inputs)
        x = self.relu(x)
        return x

class UNet_paper(torch.nn.Module):
    def __init__(self, activation = "PRelu"):
        super().__init__()
        """Encoder"""
        self.enc_conv0a = nn.Conv2d(3, 32, (3, 3), padding='same')
        self.bnlr0a = bn_lrelu(32, activation=activation)
        self.enc_conv0b = nn.Conv2d(32, 32, (3, 3), padding='same')
        self.bnlr0b = bn_lrelu(32, activation=activation)
        #Maxpool 1
        self.enc_conv1 = nn.Conv2d(32, 32, (3, 3), padding='same')
        self.bnlr1 = bn_lrelu(32, activation=activation)
        #Maxpool 2
        self.enc_conv2 = nn.Conv2d(32, 32, (3, 3), padding='same')
        self.bnlr2 = bn_lrelu(32, activation=activation)
        #Maxpool 3
        self.enc_conv3 = nn.Conv2d(32, 32, (3, 3), padding='same')
        self.bnlr3 = bn_lrelu(32, activation=activation)
        #Maxpool 4
        self.enc_conv4 = nn.Conv2d(32, 32, (3, 3), padding='same')
        self.bnlr4 = bn_lrelu(32, activation=activation)
        #Maxpool 5

        """Bottleneck"""
        self.enc_conv5 = nn.Conv2d(32, 32, (3, 3), padding='same')
        self.bnlr5 = bn_lrelu(32, activation=activation)

        """Decoder"""
        self.upsample5 = nn.UpsamplingNearest2d(scale_factor=2)
        # Concat pool 4
        self.dec_conv5a = nn.Conv2d(64, 64, (3, 3), padding='same')
        self.dec_bnlr5a = bn_lrelu(64, activation=activation)
        self.dec_conv5b = nn.Conv2d(64, 64, (3, 3), padding='same')
        self.dec_bnlr5b = bn_lrelu(64, activation=activation)
        self.upsample4 = nn.UpsamplingNearest2d(scale_factor=2)
        # Concat pool 3
        self.dec_conv4a = nn.Conv2d(96, 64, (3, 3), padding='same')
        self.dec_bnlr4a = bn_lrelu(64, activation=activation)
        self.dec_conv4b = nn.Conv2d(64, 64, (3, 3), padding='same')
        self.dec_bnlr4b = bn_lrelu(64, activation=activation)
        self.upsample3 = nn.UpsamplingNearest2d(scale_factor=2)
        # Concat pool 2
        self.dec_conv3a = nn.Conv2d(96, 64, (3, 3), padding='same')
        self.dec_bnlr3a = bn_lrelu(64, activation=activation)
        self.dec_conv3b = nn.Conv2d(64, 64, (3, 3), padding='same')
        self.dec_bnlr3b = bn_lrelu(64, activation=activation)
        self.upsample2 = nn.UpsamplingNearest2d(scale_factor=2)
        # Concat pool 1
        self.dec_conv2a = nn.Conv2d(96, 64, (3, 3), padding='same')
        self.dec_bnlr2a = bn_lrelu(64, activation=activation)
        self.dec_conv2b = nn.Conv2d(64, 64, (3, 3), padding='same')
        self.dec_bnlr2b = bn_lrelu(64, activation=activation)
        self.upsample1 = nn.UpsamplingNearest2d(scale_factor=2)
        # Concat input
        self.dec_conv1a = nn.Conv2d(64+3, 32, (3, 3), padding='same')
        self.dec_bnlr1a = bn_lrelu(32, activation=activation)
        self.dec_conv1b = nn.Conv2d(32, 16, (3, 3), padding='same')
        self.dec_bnlr1b = bn_lrelu(16, activation=activation)
        self.dec_conv1c = nn.Conv2d(16, 3, (3, 3), padding='same')

    def forward(self, x):
        """Encoder"""
        input = x.clone()
        x = self.enc_conv0a(x)
        x = self.bnlr0a(x)
        x = self.enc_conv0b(x)
        x = self.bnlr0b(x)
        pool1 = F.max_pool2d(x, 2)
        x = self.enc_conv1(pool1)
        x = self.bnlr1(x)
        pool2 = F.max_pool2d(x, 2)
        x = self.enc_conv2(pool2)
        x = self.bnlr2(x)
        pool3 = F.max_pool2d(x, 2)
        x = self.enc_conv3(pool3)
        x = self.bnlr3(x)
        pool4 = F.max_pool2d(x, 2)
        x = self.enc_conv4(pool4)
        x = self.bnlr4(x)

        """Bottleneck"""
        pool5 = F.max_pool2d(x, 2)
        x = self.enc_conv5(pool5)
        x = self.bnlr5(x)

        """Decoder"""
        x = self.upsample5(x)
        x = torch.cat([x, pool4], dim=1)
        x = self.dec_conv5a(x)
        x = self.dec_bnlr5a(x)
        x = self.dec_conv5b(x)
        x = self.dec_bnlr5b(x)
        x = self.upsample4(x)
        x = torch.cat([x, pool3], dim=1)
        x = self.dec_conv4a(x)
        x = self.dec_bnlr4a(x)
        x = self.dec_conv4b(x)
        x = self.dec_bnlr4b(x)
        x = self.upsample3(x)
        x = torch.cat([x, pool2], dim=1)
        x = self.dec_conv3a(x)
        x = self.dec_bnlr3a(x)
        x = self.dec_conv3b(x)
        x = self.dec_bnlr3b(x)
        x = self.upsample2(x)
        x = torch.cat([x, pool1], dim=1)
        x = self.dec_conv2a(x)
        x = self.dec_bnlr2a(x)
        x = self.dec_conv2b(x)
        x = self.dec_bnlr2b(x)
        x = self.upsample1(x)
        x = torch.cat([x, input], dim=1)
        x = self.dec_conv1a(x)
        x = self.dec_bnlr1a(x)
        x = self.dec_conv1b(x)
        x = self.dec_bnlr1b(x)
        x = self.dec_conv1c(x)
        return x + input

class UNet_lean(torch.nn.Module):
    def __init__(self, activation= "PRelu"):
        super().__init__()
        """Encoder"""
        self.enc_conv0a = nn.Conv2d(3, 32, (3, 3), padding='same')
        self.bnlr0a = bn_lrelu(32, activation= activation)
        self.enc_conv0b = nn.Conv2d(32, 32, (3, 3), padding='same')
        self.bnlr0b = bn_lrelu(32, activation= activation)
        # Maxpool 1
        self.enc_conv1 = nn.Conv2d(32, 32, (3, 3), padding='same')
        self.bnlr1 = bn_lrelu(32, activation= activation)
        # Maxpool 2
        self.enc_conv2 = nn.Conv2d(32, 32, (3, 3), padding='same')
        self.bnlr2 = bn_lrelu(32, activation= activation)
        # Maxpool 3

        """Bottleneck"""
        self.enc_conv3 = nn.Conv2d(32, 32, (3, 3), padding='same')
        self.bnlr3 = bn_lrelu(32, activation= activation)

        """Decoder"""
        self.upsample3 = nn.UpsamplingNearest2d(scale_factor=2)
        # Concat pool 2
        self.dec_conv3a = nn.Conv2d(64, 64, (3, 3), padding='same')
        self.dec_bnlr3a = bn_lrelu(64, activation= activation)
        self.dec_conv3b = nn.Conv2d(64, 64, (3, 3), padding='same')
        self.dec_bnlr3b = bn_lrelu(64, activation= activation)
        self.upsample2 = nn.UpsamplingNearest2d(scale_factor=2)
        # Concat pool 1
        self.dec_conv2a = nn.Conv2d(96, 64, (3, 3), padding='same')
        self.dec_bnlr2a = bn_lrelu(64, activation= activation)
        self.dec_conv2b = nn.Conv2d(64, 64, (3, 3), padding='same')
        self.dec_bnlr2b = bn_lrelu(64, activation= activation)
        self.upsample1 = nn.UpsamplingNearest2d(scale_factor=2)
        # Concat input
        self.dec_conv1a = nn.Conv2d(64 + 3, 32, (3, 3), padding='same')
        self.dec_bnlr1a = bn_lrelu(32, activation= activation)
        self.dec_conv1b = nn.Conv2d(32, 16, (3, 3), padding='same')
        self.dec_bnlr1b = bn_lrelu(16, activation= activation)
        self.dec_conv1c = nn.Conv2d(16, 3, (3, 3), padding='same')

    def forward(self, x):
        """Encoder"""
        input = x.clone()
        x = self.enc_conv0a(x)
        x = self.bnlr0a(x)
        x = self.enc_conv0b(x)
        x = self.bnlr0b(x)
        pool1 = F.max_pool2d(x, 2)
        x = self.enc_conv1(pool1)
        x = self.bnlr1(x)
        pool2 = F.max_pool2d(x, 2)
        x = self.enc_conv2(pool2)
        x = self.bnlr2(x)
        """Bottleneck"""
        pool3 = F.max_pool2d(x, 2)
        x = self.enc_conv3(pool3)
        x = self.bnlr3(x)

        """Decoder"""
        x = self.upsample3(x)
        x = torch.cat([x, pool2], dim=1)
        x = self.dec_conv3a(x)
        x = self.dec_bnlr3a(x)
        x = self.dec_conv3b(x)
        x = self.dec_bnlr3b(x)
        x = self.upsample2(x)
        x = torch.cat([x, pool1], dim=1)
        x = self.dec_conv2a(x)
        x = self.dec_bnlr2a(x)
        x = self.dec_conv2b(x)
        x = self.dec_bnlr2b(x)
        x = self.upsample1(x)
        x = torch.cat([x, input], dim=1)
        x = self.dec_conv1a(x)
        x = self.dec_bnlr1a(x)
        x = self.dec_conv1b(x)
        x = self.dec_bnlr1b(x)
        x = self.dec_conv1c(x)
        return x + input

class res_block(nn.Module):
    def __init__(self, input_ch, output_ch, activation="PRelu"):
        super().__init__()

        """Convolution"""
        self.b1 = bn_lrelu(input_ch, activation=activation)
        self.c1 = nn.Conv2d(input_ch, output_ch, (3,3), padding="same")
        self.b2 = bn_lrelu(output_ch, activation=activation)
        self.c2 = nn.Conv2d(output_ch, output_ch, (3,3), padding="same")

        """Shortcut Connection"""
        self.s = nn.Conv2d(input_ch, output_ch, (1,1), padding=0)

    def forward(self, inputs):
        x = self.b1(inputs)
        x = self.c1(x)
        x = self.b2(x)
        x - self.c2(x)
        s = self.s(inputs)
        skip = x + s
        return skip

class decoder_block(nn.Module):
    def __init__(self, input_ch, mid_channel, output_ch, activation="PRelu"):
        super().__init__()
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.r = res_block(input_ch+mid_channel, output_ch, activation=activation)

    def forward(self, inputs, skip):
        x = self.upsample(inputs)
        x = torch.cat([x, skip], dim=1)
        x = self.r(x)
        return x


class ResUnet_lean(torch.nn.Module):
    def __init__(self, activation="PRelu", encoder_ch = 32):
        super().__init__()
        """Encoder 1"""
        self.conv0a = nn.Conv2d(3, encoder_ch, (3, 3), padding='same')
        self.bnlr0a = bn_lrelu(encoder_ch, activation=activation)
        self.conv0b = nn.Conv2d(encoder_ch, encoder_ch, (3, 3), padding='same')
        self.resadd = nn.Conv2d(3, encoder_ch, (1,1), padding=0)

        """Encoder 2 & 3"""
        self.r2 = res_block(encoder_ch, encoder_ch, activation=activation)
        self.r3 = res_block(encoder_ch, encoder_ch, activation=activation)

        """Bottlneck"""
        self.r4 = res_block(encoder_ch, encoder_ch, activation=activation)

        """Decoder"""
        self.d1 = decoder_block(encoder_ch, encoder_ch, encoder_ch, activation=activation)
        self.d2 = decoder_block(encoder_ch, encoder_ch, encoder_ch, activation=activation)
        self.d3 = decoder_block(encoder_ch, 3, 16, activation=activation)

        """Output"""
        self.output = nn.Conv2d(16, 3, (3,3), padding="same")

    def forward(self, inputs):
        """HDR attempt"""
        # x = (inputs/(1+inputs))**(1/2.2)
        # x = self.conv0a(x)
        skip1 = inputs.clone()
        x = self.conv0a(inputs)
        x = self.bnlr0a(x)
        x = self.conv0b(x)
        s = self.resadd(inputs)
        x = x + s   # 8*32*32

        """Encoder 2 & 3"""
        skip2 = F.max_pool2d(x, 2) # 8*16*16
        x = self.r2(skip2)  # 16*16*16
        skip3 = F.max_pool2d(x, 2)  # 16*8*8
        x = self.r3(skip3)     # 32*8*8
        # b = self.r2(x)

        """Bottleneck"""
        x = F.max_pool2d(x, 2)  #32*4*4
        b = self.r4(x)      #64*4*4

        """Decoder"""
        d1 = self.d1(b, skip3)
        d2 = self.d2(d1, skip2)
        d3 = self.d3(d2, skip1)

        output = self.output(d3)
        return output + inputs


class ResUnet_incr_ch(torch.nn.Module):
    def __init__(self, activation="PRelu"):
        super().__init__()
        """Encoder 1"""
        self.conv0a = nn.Conv2d(3, 8, (3, 3), padding='same')
        self.bnlr0a = bn_lrelu(8, activation=activation)
        self.conv0b = nn.Conv2d(8, 8, (3, 3), padding='same')
        self.resadd = nn.Conv2d(3, 8, (1,1), padding=0)

        """Encoder 2 & 3"""
        self.r2 = res_block(8, 16, activation=activation)
        self.r3 = res_block(16, 32, activation=activation)

        """Bottlneck"""
        self.r4 = res_block(32, 64, activation=activation)

        """Decoder"""
        self.d1 = decoder_block(64, 16, 32, activation=activation)
        self.d2 = decoder_block(32, 8, 16, activation=activation)
        self.d3 = decoder_block(16, 3, 8, activation=activation)

        """Output"""
        self.output = nn.Conv2d(8, 3, (3,3), padding="same")

    def forward(self, inputs):
        """HDR attempt"""
        # x = (inputs/(1+inputs))**(1/2.2)
        # x = self.conv0a(x)
        skip1 = inputs.clone()
        x = self.conv0a(inputs)
        x = self.bnlr0a(x)
        x = self.conv0b(x)
        s = self.resadd(inputs)
        x = x + s   # 8*32*32

        """Encoder 2 & 3"""
        skip2 = F.max_pool2d(x, 2) # 8*16*16
        x = self.r2(skip2)  # 16*16*16
        skip3 = F.max_pool2d(x, 2)  # 16*8*8
        x = self.r3(skip3)     # 32*8*8
        # b = self.r2(x)

        """Bottleneck"""
        x = F.max_pool2d(x, 2)  #32*4*4
        b = self.r4(x)      #64*4*4

        """Decoder"""
        d1 = self.d1(b, skip3)
        d2 = self.d2(d1, skip2)
        d3 = self.d3(d2, skip1)

        output = self.output(d3)
        return output + inputs

class DnCNN(torch.nn.Module):
    def __init__(self, activation="PRelu", encoder_ch=32):
        super().__init__()
        """Encoder 1"""
        self.conv1 = nn.Conv2d(3, encoder_ch, (3, 3), padding='same')
        self.bnlr1 = bn_lrelu(encoder_ch, activation=activation)
        self.conv2 = nn.Conv2d(encoder_ch, encoder_ch, (3, 3), padding='same')
        self.bnlr2 = bn_lrelu(encoder_ch, activation=activation)
        self.conv3 = nn.Conv2d(encoder_ch, encoder_ch, (3, 3), padding='same')
        self.bnlr3 = bn_lrelu(encoder_ch, activation=activation)
        self.conv4 = nn.Conv2d(encoder_ch, encoder_ch, (3, 3), padding='same')
        self.bnlr4 = bn_lrelu(encoder_ch, activation=activation)
        self.conv5 = nn.Conv2d(encoder_ch, encoder_ch, (3, 3), padding='same')
        self.bnlr5 = bn_lrelu(encoder_ch, activation=activation)
        self.conv6 = nn.Conv2d(encoder_ch, encoder_ch, (3, 3), padding='same')
        self.bnlr6 = bn_lrelu(encoder_ch, activation=activation)
        self.output = nn.Conv2d(encoder_ch, 3, (3, 3), padding='same')

    def forward(self, inputs):
        """Remove the skip steps to try the simple DcCNN"""
        x = self.conv1(inputs)
        x = self.bnlr1(x)
        x = self.conv2(x)
        x = self.bnlr2(x)
        x = self.conv3(x)
        x = self.bnlr3(x)
        x = self.conv4(x)
        x = self.bnlr4(x)
        x = self.conv5(x)
        x = self.bnlr5(x)
        x = self.conv6(x)
        x = self.bnlr6(x)
        output = self.output(x)

        return output + inputs

class DnCNN_skip(torch.nn.Module):
    def __init__(self, activation="PRelu", encoder_ch = 32):
        super().__init__()
        """Encoder 1"""
        self.conv1 = nn.Conv2d(3, encoder_ch, (3, 3), padding='same')
        self.bnlr1 = bn_lrelu(encoder_ch, activation = activation)
        self.conv2 = nn.Conv2d(encoder_ch, encoder_ch, (3, 3), padding='same')
        self.bnlr2 = bn_lrelu(encoder_ch, activation = activation)
        self.conv3 = nn.Conv2d(encoder_ch, encoder_ch, (3, 3), padding='same')
        self.bnlr3 = bn_lrelu(encoder_ch, activation = activation)
        self.conv4 = nn.Conv2d(encoder_ch, encoder_ch, (3, 3), padding='same')
        self.bnlr4 = bn_lrelu(encoder_ch, activation = activation)
        self.conv5 = nn.Conv2d(encoder_ch, encoder_ch, (3, 3), padding='same')
        self.bnlr5 = bn_lrelu(encoder_ch, activation = activation)
        self.conv6 = nn.Conv2d(encoder_ch, encoder_ch, (3, 3), padding='same')
        self.bnlr6 = bn_lrelu(encoder_ch, activation = activation)
        self.output = nn.Conv2d(encoder_ch, 3, (3, 3), padding='same')

        # Try with sigmoid
        self.norm = nn.BatchNorm2d(3)

    def forward(self, inputs):
        """Remove the skip steps to try the simple DcCNN"""
        x = self.conv1(inputs)
        skip1 = self.bnlr1(x)
        x = self.conv2(skip1)
        skip2 = self.bnlr2(x)
        x = self.conv3(x)
        skip3 = self.bnlr3(x)
        x = self.conv4(x)
        x = self.bnlr4(x)
        x = x + skip3
        x = self.conv5(x)
        x = self.bnlr5(x)
        x = x + skip2
        x = self.conv6(x)
        x = self.bnlr6(x)
        x = x + skip1
        output = self.output(x)
        return output + inputs

class DnCNN_dilation(torch.nn.Module):
    def __init__(self, activation="PRelu", encoder_ch = 32):
        super().__init__()
        """Encoder 1"""
        self.conv1 = nn.Conv2d(3, encoder_ch, (3, 3), padding='same', dilation=2)
        self.bnlr1 = bn_lrelu(encoder_ch, activation=activation)
        self.conv2 = nn.Conv2d(encoder_ch, encoder_ch, (3, 3), padding='same', dilation=2)
        self.bnlr2 = bn_lrelu(encoder_ch, activation=activation)
        self.conv3 = nn.Conv2d(encoder_ch, encoder_ch, (3, 3), padding='same', dilation=2)
        self.bnlr3 = bn_lrelu(encoder_ch, activation=activation)
        self.conv4 = nn.Conv2d(encoder_ch, encoder_ch, (3, 3), padding='same', dilation=2)
        self.bnlr4 = bn_lrelu(encoder_ch, activation=activation)
        self.conv5 = nn.Conv2d(encoder_ch, encoder_ch, (3, 3), padding='same', dilation=2)
        self.bnlr5 = bn_lrelu(encoder_ch, activation=activation)
        self.conv6 = nn.Conv2d(encoder_ch, encoder_ch, (3, 3), padding='same', dilation=2)
        self.bnlr6 = bn_lrelu(encoder_ch, activation=activation)
        self.output = nn.Conv2d(encoder_ch, 3, (3, 3), padding='same', dilation=2)

    def forward(self, inputs):
        """Remove the skip steps to try the simple DcCNN"""
        x = self.conv1(inputs)
        skip1 = self.bnlr1(x)
        x = self.conv2(skip1)
        skip2 = self.bnlr2(x)
        x = self.conv3(x)
        skip3 = self.bnlr3(x)
        x = self.conv4(x)
        x = self.bnlr4(x)
        x = x + skip3
        x = self.conv5(x)
        x = self.bnlr5(x)
        x = x + skip2
        x = self.conv6(x)
        x = self.bnlr6(x)
        x = x + skip1
        output = self.output(x)

        return output + inputs

class ParallelUNet(torch.nn.Module):
    def __init__(self, activation="PRelu", encoder_ch = 32):
        super().__init__()
        self.path1 = DnCNN_dilation(activation=activation, encoder_ch=encoder_ch)
        self.path2 = ResUnet_incr_ch(activation=activation)
        self.output = nn.Conv2d(6, 3, 1, padding="same")

    def forward(self, inputs):
        path1 = self.path1(inputs)
        path2 = self.path2(inputs)
        output = self.output(torch.cat([path1, path2], dim=1))

        return output + inputs

