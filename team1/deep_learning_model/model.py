import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import *


class mymodel(nn.Module):
    def __init__(self, emb_dim, n_hidden, n_classes=4):
        super(mymodel,self).__init__()
        self.convs = nn.Sequential(
            nn.Conv1d(emb_dim, n_hidden, 3, padding=3),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(n_hidden, n_hidden, 3, padding=3),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(n_hidden, n_hidden*2, 3, padding=3),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(n_hidden*2, n_hidden*2, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Conv1d(n_hidden*2, n_hidden*3, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )

        self.linear = nn.Sequential(
            nn.Linear(n_hidden*6, n_hidden*4),
            nn.ReLU(),
            nn.Linear(n_hidden * 4, n_hidden * 4),
            nn.ReLU(),
            nn.Linear(n_hidden * 4, n_hidden * 4),
            nn.ReLU(),
            nn.Linear(n_hidden * 4, n_classes),
        )

    def forward(self, head, body):
        head_conv = self.convs(head)
        body_conv = self.convs(body)

        head_conv, _ = torch.max(head_conv, dim=-1)
        body_conv, _ = torch.max(body_conv, dim=-1)
        feature = torch.cat([head_conv, body_conv], dim=1)

        out = self.linear(feature)
        return out

if __name__ == '__main__':

    # v = GoogleVec()
    # v.load()
    # dir = '../../../fnc-1/'
    # t0=time.time()
    # train_set = News(stances_path=os.path.join(dir, 'train_stances.csv'),
    #                bodies_path=os.path.join(dir, 'train_bodies.csv'), vecs=v)
    # print(time.time() - t0)
    # train_loader = DataLoader(train_set, batch_size=2)
    # head, body, stance = next(iter(train_loader))
    head = torch.rand(2,300,190)
    body = torch.rand(2, 300, 2000)
    model = mymodel(300,256)

    out = model(head, body)
    pred = out.argmax(1)
    print(out,pred)
