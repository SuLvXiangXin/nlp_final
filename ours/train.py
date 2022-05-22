import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data_loader import MyDataSet
import transformers
from model import Mymodel
from utils import WeightedLoss, score
import argparse
transformers.logging.set_verbosity_error()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(0)
parser = argparse.ArgumentParser()
parser.add_argument('-w', help='weight', type=str, default='0.5,1')
parser.add_argument('-p', help='experiment path', type=str, default='run/weight_0.5_1')
parser.add_argument('-b', help='batch size', type=int, default=24)
parser.add_argument('-v', help='validation size', type=float, default=0.1)
parser.add_argument('-d', help='device', type=int, default=1)
args = parser.parse_args()
exp_path = args.p
batch_size = args.b
val_size = args.v
device = 'cuda:%d'%args.d
weight = [float(w) for w in args.w.split(',')]

os.makedirs(exp_path, exist_ok=True)
train_dataset = MyDataSet("train_merged.csv")
train_set, val_set = torch.utils.data.random_split(train_dataset, [len(train_dataset) - int(len(train_dataset) * val_size),
                                                                   int(len(train_dataset) * val_size)])
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False)

model = Mymodel().to(device)
loss_func = WeightedLoss(weight)
optimizer = optim.Adam([
    {'params': model.bert.parameters(), 'lr': 1e-5},
    {'params': model.linear.parameters(), 'lr': 1e-3}
], lr=1e-5)

file = os.path.join(exp_path, 'best.pth')
if os.path.exists(file):
    print('loading checkpoint at', file)
    checkpoint = torch.load(file)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optim'])
    start_epoch = checkpoint['epoch']
    best_score = checkpoint['best_score']
else:
    start_epoch = 0
    best_score = 0
epochs = 10
best_epoch = 0
for epoch in range(start_epoch, epochs):
    loss_sum = 0
    acc_sum = 0
    score_sum = 0
    score_all = 0
    model.train()
    for i, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        stance = batch['stance'].to(device)
        optimizer.zero_grad()
        output = model(input_ids, attention_mask, token_type_ids)
        loss = loss_func(output, stance)
        pred = output.argmax(1)
        acc = (pred == stance).float().mean()
        loss.backward()
        loss_sum += loss.item()
        acc_sum += acc.item()
        score_sum += score(pred, stance)
        score_all += score(stance, stance)
        optimizer.step()
        if (i + 1) % 100 == 0:
            print(
                f'Epoch {epoch + 1}/{epochs}, iter {i + 1}/{len(train_loader)}, loss {loss}({loss_sum / (i + 1)}),'
                f' acc: {acc}({acc_sum / (i + 1)}), score: {score_sum}/{score_all}')

    loss_sum_val = 0
    acc_sum_val = 0
    score_sum_val = 0
    score_all_val = 0
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            input_ids = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            stance = batch['stance'].to(device)
            output = model(input_ids, attention_mask, token_type_ids)
            loss = loss_func(output, stance)
            pred = output.argmax(1)
            acc = (pred == stance).float().mean()
            loss_sum_val += loss.item()
            acc_sum_val += acc.item()
            score_sum_val += score(pred, stance)
            score_all_val += score(stance, stance)
    print(f'Epoch {epoch + 1}/{epochs}, val: loss {loss_sum_val / len(val_loader)}, '
          f'acc: {acc_sum_val / len(val_loader)}, score: {score_sum_val}/{score_all_val}')

    torch.save(
        {'model': model.state_dict(), 'optim': optimizer.state_dict(), 'epoch': epoch + 1, 'best_score': best_score},
        os.path.join(exp_path, '%02d.pth' % epoch))
    if score_sum_val > best_score:
        best_score = score_sum_val
        best_epoch = epoch + 1
        torch.save({'model': model.state_dict(), 'optim': optimizer.state_dict(), 'epoch': epoch, 'best_score': best_score},
                   os.path.join(exp_path, 'best.pth'))
if best_epoch > 0:
    print('The best model is at epoch %d' % best_epoch, best_score)
