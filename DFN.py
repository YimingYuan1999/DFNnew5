'''
Gelu activation
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class gelu(nn.Module):
    def __init__(self):
        super(gelu, self).__init__()
        self.pi = torch.tensor(np.pi)
        #self.pi=torch.tensor(np.pi).cuda()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(2 / self.pi) * (x + 0.044715 * torch.pow(x, 3))))

def default_conv(nc):
    return nn.Conv2d(nc, nc, kernel_size=3, stride=1, padding=1, bias=True)


class ResidualBlock(nn.Module):
    def __init__(
            self, n_feats):
        super(ResidualBlock, self).__init__()

        self.conv1 = default_conv(n_feats)
        self.conv2 = default_conv(n_feats)
        self.conv3 = default_conv(n_feats)
        self.relu1 = nn.PReLU(n_feats, 0.25)
        self.relu2 = nn.PReLU(n_feats, 0.25)
        self.relu3 = nn.PReLU(n_feats, 0.25)
        self.scale1 = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)
        self.scale2 = nn.Parameter(torch.FloatTensor([2.0]), requires_grad=True)
        self.scale3 = nn.Parameter(torch.FloatTensor([-1.0]), requires_grad=True)
        self.scale4 = nn.Parameter(torch.FloatTensor([4.0]), requires_grad=True)
        self.scale5 = nn.Parameter(torch.FloatTensor([1 / 6]), requires_grad=True)

    def forward(self, x):
        yn = x
        k1 = self.relu1(x)
        k1 = self.conv1(k1)
        yn_1 = k1 * self.scale1 + yn
        k2 = self.relu2(yn_1)
        k2 = self.conv2(k2)
        yn_2 = yn + self.scale2 * k2
        yn_2 = yn_2 + k1 * self.scale3
        k3 = self.relu3(yn_2)
        k3 = self.conv3(k3)
        yn_3 = k3 + k2 * self.scale4 + k1
        yn_3 = yn_3 * self.scale5
        out = yn_3 + yn
        return out


"""
class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = default_conv( channels)
        self.conv2 = default_conv( channels)
        self.relu = nn.PReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out) 
        out = torch.add(out, residual)
        #out1 = self.relu(self.conv1(out))
        #out1 = self.conv2(out1) 
        #out1 = torch.add(out1, residual)
        #out2 = self.relu(self.conv1(out1))
        #out2 = self.conv2(out2) 
        #out2 = torch.add(out2, residual)
        #out3 = self.relu(self.conv1(out2))
        #out3 = self.conv2(out3) 
        #out3 = torch.add(out3, residual)     
        return out
"""


class ResNet(torch.nn.Module):
    def __init__(self,channels):
        super(ResNet,self).__init__()
        self.ResidualBlock1=ResidualBlock(channels)
        self.ResidualBlock2=ResidualBlock(channels)
        self.ResidualBlock3=ResidualBlock(channels)
        self.ResidualBlock4=ResidualBlock(channels)
        self.ResidualBlock5=ResidualBlock(channels)
        
    def forward(self, x):
        out=self.ResidualBlock1(x)
        out1=self.ResidualBlock2(out)
        out1 = torch.add(out1, x)
        out2=self.ResidualBlock3(out1)
        out3=self.ResidualBlock4(out2)
        out3 = torch.add(out3, x)
        out4=self.ResidualBlock5(out3)
        return out4

class Feedbackblock(nn.Module):
    def __init__(self, num_features):
        super(Feedbackblock, self).__init__()
        self.feature_extract = ResNet(num_features)
        self.should_reset = True
        self.last_hidden = None
        self.number=-1
        self.last_total=None

    def reset_state(self):
        self.should_reset = True

    def forward(self, x):
        self.number+=1
        if self.should_reset:      #从头开始迭代
            self.last_hidden = torch.zeros(x.size()).cuda()
            self.last_hidden.copy_(x)    #copy一份x的值作为初始值传输给末端层
            self.last_total = torch.zeros(x.size()).cuda()
            self.last_total.copy_(x)
            t = torch.zeros(x.size()).cuda()
            t.copy_(self.last_total)
            x = self.feature_extract(t)
            self.last_hidden=x
            self.should_reset = False
            return x
            
        t = self.last_total / self.number     #和迭代后的值取平均
        self.last_total+=self.last_hidden
        
        x = self.feature_extract(self.last_hidden)
        
        self.last_hidden=x
        
        return x


class RRFBN(nn.Module):
    def __init__(self, iterations=7,blocks=2):
        super(RRFBN, self).__init__()
        self.iterations = iterations
        self.block_nums=blocks
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
        )

        self.output_layer = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
        )
        self.gelu = gelu()
        self.FB=nn.Sequential(*[Feedbackblock(32) for _ in range(self.block_nums)])
    def all_reset_state(self):
        for block in self.FB:
            block.reset_state()
    def forward(self, input):
        x_list = []
        self.all_reset_state()
        for idx in range(self.iterations):
            x = self.gelu(self.input_layer(input))
            x = self.FB(x)
            x = self.output_layer(x)
            x = x + input
            x_list.append(x)
        return x, x_list