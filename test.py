import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from DATA import Dataset
from net_loss import MMDNet, SNet
import numpy as np
import os
from scipy.io import savemat
import warnings
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
device = torch.device("cuda:0")
warnings.filterwarnings("ignore")


def Ttest(model):
    model.eval()
    pretext_model = torch.load('loss_mdl/4with_out_1_2/5/best.mdl')
    model2_dict = model.state_dict()
    state_dict = {k: v for k, v in pretext_model.items() if k in model2_dict.keys()}
    model2_dict.update(state_dict)
    model.load_state_dict(model2_dict)
    path = '/run/media/xd132/F/ZTZ/YYF/'
    db = Dataset(path, 'data/BAR/Q1.mat', 'data/BAR/Q2.mat', 'data/BAR/REF.mat', 'data/BAY/Q1.mat', 'data/BAY/Q2.mat', 'data/BAY/REF.mat', 'Ttest', 224)
    test_loader = DataLoader(db, batch_size=512, shuffle=False)
    if not os.path.exists('result'):
        os.makedirs('result')
    output = np.zeros((984, 740))
    with torch.no_grad():
        for step, (image_1, image_2, h, w) in enumerate(test_loader):
            image_1 = image_1.type(torch.float32).to(device)
            image_2 = image_2.type(torch.float32).to(device)
            torch.cuda.synchronize()
            Sout = model(image_1, image_2)
            output[h.numpy(), w.numpy()] = Sout.argmax(dim=1).detach().cpu().numpy() + 1
            print(step)
    filename = "result//T.mat"
    savemat(filename, {"data": output})

def TFtest(model):
    model.eval()
    pretext_model = torch.load('loss_mdl/4with_out_1_2/20/best.mdl')
    model2_dict = model.state_dict()
    state_dict = {k: v for k, v in pretext_model.items() if k in model2_dict.keys()}
    model2_dict.update(state_dict)
    model.load_state_dict(model2_dict)
    path = '/run/media/xd132/F/ZTZ/YYF/'
    db = Dataset(path, 'data/BAR/Q1.mat', 'data/BAR/Q2.mat', 'data/BAR/REF.mat', 'data/BAY/Q1.mat', 'data/BAY/Q2.mat',
                 'data/BAY/REF.mat', 'Ttest', 224)
    batch = 512
    test_loader = DataLoader(db, batch_size=batch, shuffle=False)
    if not os.path.exists('result'):
        os.makedirs('result')
    output = np.zeros((984,740,2))
    count = 0

    with torch.no_grad():
        for step, (image_1, image_2, h, w) in enumerate(test_loader):
            image_1 = image_1.type(torch.float32).to(device)
            image_2 = image_2.type(torch.float32).to(device)
            torch.cuda.synchronize()
            Sout = model(image_1, image_2)
            output[h.numpy(),w.numpy(),:] = Sout.detach().cpu().numpy()
            print(step)
        filename = "result//TF.mat"
    savemat(filename, {"data": output})

def Alltest(model):
    model.eval()
    pretext_model = torch.load('percent/20/best500.mdl')
    model2_dict = model.state_dict()
    state_dict = {k: v for k, v in pretext_model.items() if k in model2_dict.keys()}
    model2_dict.update(state_dict)
    model.load_state_dict(model2_dict)
    path = '/run/media/xd132/F/ZTZ/YYF/'
    db = Dataset(path, 'data/BAR/Q1.mat', 'data/BAR/Q2.mat', 'data/BAR/REF.mat', 'data/BAY/Q1.mat', 'data/BAY/Q2.mat', 'data/BAY/REF.mat', 'Ttest', 224)
    batch = 128
    test_loader = DataLoader(db, batch_size=128, shuffle=False)
    if not os.path.exists('result'):
        os.makedirs('result')
    S9 = np.zeros((73481, 32))
    S5 = np.zeros((73481, 32))
    S3 = np.zeros((73481, 32))
    T9 = np.zeros((132552, 32))
    T5 = np.zeros((132552, 32))
    T3 = np.zeros((132552, 32))
    count = 0
    with torch.no_grad():
        for step, (T1, T2, _, _) in enumerate(test_loader):
            T1 = T1.type(torch.float32).to(device)
            T2 = T2.type(torch.float32).to(device)
            torch.cuda.synchronize()
            _, Tout9, Tout5, Tout3 = model(T1, T2)
            T9[count:(count + batch), :] = Tout9.detach().cpu().numpy()
            T5[ count:(count + batch), :] = Tout5.detach().cpu().numpy()
            T3[count:(count + batch), :] = Tout3.detach().cpu().numpy()
            count += batch
            print(step)
    db = Dataset(path, 'data/BAR/Q1.mat', 'data/BAR/Q2.mat', 'data/BAR/REF.mat', 'data/BAY/Q1.mat', 'data/BAY/Q2.mat', 'data/BAY/REF.mat', 'Stest', 224)
    batch = 128
    test_loader = DataLoader(db, batch_size=128, shuffle=False)
    count = 0
    with torch.no_grad():
        for step, (S1, S2, _, _) in enumerate(test_loader):
            S1 = S1.type(torch.float32).to(device)
            S2 = S2.type(torch.float32).to(device)
            torch.cuda.synchronize()
            _, Sout9, Sout5, Sout3 = model(S1, S2)
            S9[count:(count + batch), :] = Sout9.detach().cpu().numpy()
            S5[count:(count + batch), :] = Sout5.detach().cpu().numpy()
            S3[count:(count + batch), :] = Sout3.detach().cpu().numpy()
            count += batch
            print(step)
    savemat("result//S9.mat", {"data": S9})
    savemat("result//S5.mat", {"data": S5})
    savemat("result//S3.mat", {"data": S3})
    savemat("result//T9.mat", {"data": T9})
    savemat("result//T5.mat", {"data": T5})
    savemat("result//T3.mat", {"data": T9})

def ATtest(model):
    path = '/run/media/xd132/F/ZTZ/YYF/'
    db = Dataset(path, 'data/BAR/Q1.mat', 'data/BAR/Q2.mat', 'data/BAR/REF.mat', 'data/BAY/Q1.mat', 'data/BAY/Q2.mat', 'data/BAY/REF.mat', 'Ttest', 224)
    test_loader = DataLoader(db, batch_size=512, shuffle=False)
    if not os.path.exists('result'):
        os.makedirs('result')
    model.eval()

    for i in range(10, 990, 10):
        print('ablation/10/20/best{}.mdl'.format(str(i)))
        pretext_model = torch.load('ablation/2.5/5/best{}.mdl'.format(str(i)))
        model2_dict = model.state_dict()
        state_dict = {k: v for k, v in pretext_model.items() if k in model2_dict.keys()}
        model2_dict.update(state_dict)
        model.load_state_dict(model2_dict)
        output = np.zeros((984, 740))
        with torch.no_grad():
            for step, (image_1, image_2, h, w) in enumerate(test_loader):
                image_1 = image_1.type(torch.float32).to(device)
                image_2 = image_2.type(torch.float32).to(device)
                torch.cuda.synchronize()
                Sout, _, _, _ = model(image_1, image_2)
                output[h.numpy(), w.numpy()] = Sout.argmax(dim=1).detach().cpu().numpy() + 1
        filename = "result//T{}.mat".format(str(i))
        savemat(filename, {"data": output})


if __name__ == "__main__":
    model = SNet(224).to(device)
    model = nn.DataParallel(model)
    Ttest(model)