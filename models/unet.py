import torch
import torch.nn as nn
import torchvision



# Define network
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x  
      
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.in_channels = in_channels
        
        # Following Decoder Block in Figure 2
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            ConvBlock(out_channels,out_channels),
            ConvBlock(out_channels,out_channels)
        )



    def forward(self, x):
        return self.block(x)
      
class UNet(nn.Module):

    def __init__(self, n_classes=1, num_filters=32, pretrained=True, is_deconv=False, dropout_p=0.3):
        """
        :param n_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network is used
            True  - encoder is pre-trained with resnet34
        :is_deconv:
            False: bilinear interpolation is used in decoder
            True: deconvolution is used in decoder
        """
        super().__init__()
        self.n_classes = n_classes
        self.pool = nn.MaxPool2d(2, 2)
        self.encoder = torchvision.models.resnet34(pretrained=pretrained)
#         self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)
        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4
        
        self.center = DecoderBlock(512, 256) # self.center is the decoder right after the pool 2x2
        
        self.dec5 = DecoderBlock(768, 256)
        self.dec4 = DecoderBlock(512, 256)
        self.dec3 = DecoderBlock(384, 64)
        self.dec2 = DecoderBlock(128, 128)
        self.dec1 = DecoderBlock(128, 32)
        self.dec0 = ConvBlock(32, 32)
        self.final = nn.Conv2d(32, n_classes, kernel_size=1)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))

        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1)

        if self.n_classes > 1:
            x_out = F.log_softmax(self.final(dec0), dim=1)
        else:
            x_out = torch.sigmoid(self.final(dec0))

#         return self.dropout(x_out)
        return x_out