import torch
import torch.nn as nn

'''
全连接设置为3层
'''

class Attention(nn.Module):
    def __init__(self, channel, ratio=16):
        super(Attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        a = self.sigmoid(avgout + maxout)
        out = x*a
        return out


class Ex_Net(nn.Module):
    def __init__(self, Cin):
        super(Ex_Net, self).__init__()
        self.relu = nn.PReLU()
        self.conv1 = nn.Conv2d(Cin, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.maxpool1 = nn.AdaptiveAvgPool2d((5, 5))

        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, 1)
        self.maxpool2 = nn.AdaptiveAvgPool2d((3, 3))

        self.conv5 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv6 = nn.Conv2d(128, 128, 3, 1, 1)

    def forward(self, x):

        out = self.relu(self.conv1(x))
        out_9 = self.relu(self.conv2(out))
        out = self.relu(self.maxpool1(out_9))
        out = self.relu(self.conv3(out))
        out_5 = self.relu(self.conv4(out))
        out = self.relu(self.maxpool2(out_5))
        out = self.relu(self.conv5(out))
        out_3 = self.conv6(out)

        return out_9, out_5, out_3


class CNet(nn.Module):
    def __init__(self, Cin):
        super(CNet, self).__init__()
        self.cnn1 = Ex_Net(Cin)
        self.cnn2 = Ex_Net(Cin)

        self.SA9 = Attention(64)
        self.SA5 = Attention(128)
        self.SA3 = Attention(128)

        self.relu = nn.PReLU()
        self.linear91 = nn.Linear(64*9*9, 512)
        self.linear51 = nn.Linear(128*5*5, 512)
        self.linear31 = nn.Linear(128*3*3, 512)
        self.linear92 = nn.Linear(512, 128)
        self.linear52 = nn.Linear(512, 128)
        self.linear32 = nn.Linear(512, 128)


    def forward(self, T1, T2):
        T19, T15, T13 = self.cnn1(T1)
        T29, T25, T23 = self.cnn2(T2)
        b, _, _, _ = T1.shape
        out9 = abs(self.SA9(T19 - T29).reshape(b,-1))
        out5 = abs(self.SA5(T15 - T25).reshape(b, -1))
        out3 = abs(self.SA3(T13 - T23).reshape(b, -1))
        out9 = self.relu(self.linear91(out9))
        out5 = self.relu(self.linear51(out5))
        out3 = self.relu(self.linear31(out3))
        out9 = self.relu(self.linear92(out9))
        out5 = self.relu(self.linear52(out5))
        out3 = self.relu(self.linear32(out3))

        return out9, out5, out3



class MMDNet(nn.Module):
    def __init__(self, Cin):
        super(MMDNet, self).__init__()
        self.SCNet = CNet(Cin)

        self.SCL91 = nn.Linear(128, 64)
        self.SCL51 = nn.Linear(128, 64)
        self.SCL31 = nn.Linear(128, 64)
        self.SCL92 = nn.Linear(64, 32)
        self.SCL52 = nn.Linear(64, 32)
        self.SCL32 = nn.Linear(64, 32)
        self.SCL93 = nn.Linear(32, 2)
        self.SCL53 = nn.Linear(32, 2)
        self.SCL33 = nn.Linear(32, 2)
        self.lamba1 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.lamba2 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.lamba3 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.lamba1.data.fill_(0.33)
        self.lamba2.data.fill_(0.33)
        self.lamba3.data.fill_(0.33)
        self.relu = nn.PReLU()
    def forward(self, S1, S2, T1, T2):
        Sout9, Sout5, Sout3 = self.SCNet(S1, S2)
        Tout9, Tout5, Tout3 = self.SCNet(T1, T2)

        Sout9 = self.relu(self.SCL91(Sout9))
        Sout5 = self.relu(self.SCL51(Sout5))
        Sout3 = self.relu(self.SCL31(Sout3))
        Sout9 = self.relu(self.SCL92(Sout9))
        Sout5 = self.relu(self.SCL52(Sout5))
        Sout3 = self.relu(self.SCL32(Sout3))
        SR_9 = self.SCL93(Sout9)
        SR_5 = self.SCL53(Sout5)
        SR_3 = self.SCL33(Sout3)
        SCF = self.lamba1 * SR_9 + self.lamba2 * SR_5 + self.lamba3 * SR_3

        Tout9 = self.relu(self.SCL91(Tout9))
        Tout5 = self.relu(self.SCL51(Tout5))
        Tout3 = self.relu(self.SCL31(Tout3))
        Tout9 = self.relu(self.SCL92(Tout9))
        Tout5 = self.relu(self.SCL52(Tout5))
        Tout3 = self.relu(self.SCL32(Tout3))
        TR_9 = self.SCL93(Tout9)
        TR_5 = self.SCL53(Tout5)
        TR_3 = self.SCL33(Tout3)

        TCF = self.lamba1 * TR_9 + self.lamba2 * TR_5 + self.lamba3 * TR_3

        return SCF, SR_9, SR_5, SR_3, Sout9, Sout5, Sout3, TCF, TR_9, TR_5, TR_3, Tout9, Tout5, Tout3
        # return SCF
class SNet(nn.Module):
    def __init__(self, Cin):
        super(SNet, self).__init__()
        self.SCNet = CNet(Cin)

        self.SCL91 = nn.Linear(128, 64)
        self.SCL51 = nn.Linear(128, 64)
        self.SCL31 = nn.Linear(128, 64)
        self.SCL92 = nn.Linear(64, 32)
        self.SCL52 = nn.Linear(64, 32)
        self.SCL32 = nn.Linear(64, 32)
        self.SCL93 = nn.Linear(32, 2)
        self.SCL53 = nn.Linear(32, 2)
        self.SCL33 = nn.Linear(32, 2)
        self.lamba1 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.lamba2 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.lamba3 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.relu = nn.PReLU()
    def forward(self, T1, T2):
        Tout9, Tout5, Tout3 = self.SCNet(T1, T2)

        Tout9 = self.relu(self.SCL91(Tout9))
        Tout5 = self.relu(self.SCL51(Tout5))
        Tout3 = self.relu(self.SCL31(Tout3))
        Tout9 = self.relu(self.SCL92(Tout9))
        Tout5 = self.relu(self.SCL52(Tout5))
        Tout3 = self.relu(self.SCL32(Tout3))
        TR_9 = self.SCL93(Tout9)
        TR_5 = self.SCL53(Tout5)
        TR_3 = self.SCL33(Tout3)
        TCF = self.lamba1 * TR_9 + self.lamba2 * TR_5 + self.lamba3 * TR_3

        return TCF, Tout9, Tout5, Tout3


