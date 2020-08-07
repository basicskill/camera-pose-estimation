import torch
from torch import nn

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        alexnet = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
        
        # Convolutional layers from AlexNet
        self.conv1 = alexnet.features
        self.conv2 = alexnet.features # TODO: Does it work??

        self.conv1.requires_grad = False
        self.conv2.requires_grad = False

        self.pyramid = [
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.AdaptiveMaxPool2d((2, 2)),
            nn.AdaptiveMaxPool2d((3, 3)),
            nn.AdaptiveMaxPool2d((6, 6)),
            # nn.AdaptiveMaxPool2d((13, 13))
        ]

        self.fcQ = nn.Linear(25600, 4)
        self.fcT = nn.Linear(25600, 3)


    def forward(self, first, second):

        out1 = self.conv1(first)
        out2 = self.conv2(second)

        spp1 = torch.cat([ pool(out1).flatten() for pool in self.pyramid], dim=0)
        spp2 = torch.cat([ pool(out2).flatten() for pool in self.pyramid], dim=0)

        spp = torch.cat((spp1, spp2), dim=0)

        t = self.fcT(spp)
        q = self.fcQ(spp)

        return t, q