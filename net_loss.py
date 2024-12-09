import torch.nn as nn

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

        return out_3

class CNet(nn.Module):
    def __init__(self, Cin):
        super(CNet, self).__init__()
        self.cnn1 = Ex_Net(Cin)
        self.cnn2 = Ex_Net(Cin)
        self.SA3 = Attention(128)
        self.relu = nn.PReLU()
        self.linear31 = nn.Linear(128*3*3, 512)
        self.linear32 = nn.Linear(512, 128)

    def forward(self, T1, T2):
        T13 = self.cnn1(T1)
        T23 = self.cnn2(T2)
        b, _, _, _ = T1.shape
        out3 = abs(self.SA3(T13 - T23).reshape(b, -1))
        out3 = self.relu(self.linear31(out3))
        out3 = self.relu(self.linear32(out3))

        return out3


class MMDNet(nn.Module):
    def __init__(self, Cin):
        super(MMDNet, self).__init__()
        self.SCNet = CNet(Cin)
        self.SCL31 = nn.Linear(128, 64)
        self.SCL32 = nn.Linear(64, 32)
        self.SCL33 = nn.Linear(32, 2)
        self.relu = nn.PReLU()

    def forward(self, S1, S2, T1, T2):
        Sout3 = self.SCNet(S1, S2)
        Tout3 = self.SCNet(T1, T2)

        Sout3 = self.relu(self.SCL31(Sout3))
        Sout3 = self.relu(self.SCL32(Sout3))
        SR_3 = self.SCL33(Sout3)
        SCF = SR_3
        Tout3 = self.relu(self.SCL31(Tout3))
        Tout3 = self.relu(self.SCL32(Tout3))
        TR_3 = self.SCL33(Tout3)
        TCF = TR_3

        return SCF, Sout3, TCF, Tout3


class SNet(nn.Module):
    def __init__(self, Cin):
        super(SNet, self).__init__()
        self.SCNet = CNet(Cin)
        self.SCL31 = nn.Linear(128, 64)
        self.SCL32 = nn.Linear(64, 32)
        self.SCL33 = nn.Linear(32, 2)
        self.relu = nn.PReLU()

    def forward(self, T1, T2):

        Tout3 = self.SCNet(T1, T2)
        Tout3 = self.relu(self.SCL31(Tout3))
        Tout3 = self.relu(self.SCL32(Tout3))
        TR_3 = self.SCL33(Tout3)
        TCF = TR_3

        return TCF


