import torch
from torch import nn
from copy import deepcopy

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        alexnet = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
        
        # Convolutional layers from AlexNet
        self.conv1 = deepcopy(alexnet.features)
        self.conv2 = deepcopy(alexnet.features)

        self.conv1.requires_grad = False
        self.conv2.requires_grad = False

        self.pyramid = [
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.AdaptiveMaxPool2d((2, 2)),
            nn.AdaptiveMaxPool2d((3, 3)),
            nn.AdaptiveMaxPool2d((6, 6)),
            # nn.AdaptiveMaxPool2d((13, 13))
        ]

        self.fcQ = nn.Linear(25600, 3)
        self.fcT = nn.Linear(25600, 3)

        self.beta = 10

    def forward(self, first, second):

        out1 = self.conv1(first)
        out2 = self.conv2(second)

        spp1 = torch.cat([ torch.flatten(pool(out1), start_dim=1) for pool in self.pyramid ], dim=1)
        spp2 = torch.cat([ torch.flatten(pool(out2), start_dim=1) for pool in self.pyramid ], dim=1)

        spp = torch.cat((spp1, spp2), dim=1)

        t = self.fcT(spp)
        # TODO: Modifikovati za Ojlerove uglove
        q = self.fcQ(spp)

        return t, q
    
    def loss(self, t, q, t_true, q_true):
        out_loss = torch.norm(t - t_true, dim=1) + self.beta * torch.norm(q - q_true, dim=1)
        return out_loss.mean()