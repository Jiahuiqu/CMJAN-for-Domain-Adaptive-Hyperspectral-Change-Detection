import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from Data_set import Dataset
from net import MMDNet
import warnings

warnings.filterwarnings("ignore")
'''多尺度90的结果，还待调整'''
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
device = torch.device("cuda:0")

class MMD_loss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            with torch.no_grad():
                XX = torch.mean(kernels[:batch_size, :batch_size])
                YY = torch.mean(kernels[batch_size:, batch_size:])
                XY = torch.mean(kernels[:batch_size, batch_size:])
                YX = torch.mean(kernels[batch_size:, :batch_size])
                loss = torch.mean(XX + YY - XY - YX)
                del XX, YY, XY, YX
            torch.cuda.empty_cache()
            return float(loss)
def adjust_learning_rate(lr, optimizer):
    """Sets the learning rate to the initial LR decayed by 10 every 20 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train_epoch(model, optimizer, criteron, train_data):
    model.train()
    mmd = MMD_loss(kernel_type='linear')
    loss_meter, loss_meter1, loss_meter2, loss_meter3, count_it = 0, 0, 0, 0, 0
    for step, ( S1, S2, REFS, T1, T2, _) in enumerate(train_data):
        S1 = S1.type(torch.float32).to(device)
        S2 = S2.type(torch.float32).to(device)
        T1 = T1.type(torch.float32).to(device)
        T2 = T2.type(torch.float32).to(device)
        label = REFS.type(torch.float32).to(device)
        SCF, SR_9, SR_5, SR_3, Sout9, Sout5, Sout3, TCF, TR_9, TR_5, TR_3, Tout9, Tout5, Tout3 = model(S1, S2, T1, T2)
        mmd_loss = 0.5 * mmd(Sout3, Tout3)
        s_loss = criteron(SR_3, label.long())
        t_loss = criteron(TR_3, label.long())
        loss = s_loss + mmd_loss + t_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_meter += loss
        loss_meter1 += mmd_loss
        loss_meter2 += s_loss
        loss_meter3 += t_loss
        count_it += 1

    return float(loss_meter / count_it), float(loss_meter1 / count_it), float(loss_meter2 / count_it), float(loss_meter3 / count_it)



def train(max_epoch, batchsz, lr, model):

    path = '/run/media/xd132/F/ZTZ/YYF/'
    db = Dataset(path, 'data/BAR/Q1.mat', 'data/BAR/Q2.mat', 'data/BAR/SBAR10.mat', 'data/BAY/Q1.mat', 'data/BAY/Q2.mat',
                 'data/BAY/REF.mat', 'train', 224)
    train_data = DataLoader(db, batch_size = batchsz, shuffle = True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteron = nn.CrossEntropyLoss()
    best_loss = 1000000000
    for epoch in range(max_epoch):
        train_loss, mmd_loss, s_loss, t_loss = train_epoch(model, optimizer, criteron, train_data)
        if epoch % 1 == 0:
            print("epoch:",epoch, best_loss, train_loss, mmd_loss, s_loss, t_loss)
            if train_loss <= best_loss:
                best_loss = train_loss
                torch.save(model.state_dict(), 'loss_mdl/with_out_1_2/15/best.mdl')
        if epoch % 10 == 0:
            torch.save(torch.load('loss_mdl/with_out_1_2/15/best.mdl'), 'loss_mdl/with_out_1_2/15/best%d.mdl' % (epoch))

        if (epoch + 1) - 100 == 0:
            lr /= 2
            adjust_learning_rate(lr, optimizer)


if __name__ == "__main__":
    model = MMDNet(224).to(device)
    model = nn.DataParallel(model)
    train(1000, 512, 0.00005,model)