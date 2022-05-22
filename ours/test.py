import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from data_loader import MyDataSet
import transformers
from model import Mymodel
from utils import score
import argparse
transformers.logging.set_verbosity_error()

parser = argparse.ArgumentParser()
parser.add_argument('-p', help='experiment path', type=str, default='run/weight_0.5_1')
args = parser.parse_args()
exp_path = args.p
test_file = pd.read_csv('../fnc-1/competition_test_stances_unlabeled.csv')
LABELS = ['agree', 'disagree', 'discuss', 'unrelated']
batch_size = 32
test_dataset = MyDataSet("test_merged.csv")
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

device = 'cuda:2'
model = Mymodel().to(device)
file = os.path.join(exp_path, 'best.pth')
if os.path.exists(file):
    print(file)
    checkpoint = torch.load(file)
    model.load_state_dict(checkpoint['model'])
else:
    raise "No checkpoint"

acc_sum = 0
score_sum = 0
score_all = 0
model.eval()
preds = []
with torch.no_grad():
    for i, batch in enumerate(test_loader):
        if (i+1)%100==0:
            print(i+1,'/',len(test_loader))
        input_ids = batch['input_ids'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        stance = batch['stance'].to(device)
        output = model(input_ids, attention_mask, token_type_ids)
        pred = output.argmax(1)
        preds.append(pred)
        acc = (pred == stance).float().mean()
        acc_sum += acc.item()
        score_sum += score(pred, stance)
        score_all += score(stance, stance)
    acc_sum /= len(test_loader)

preds = torch.cat(preds)
test_file['Stance'] = pd.Series(preds.cpu()).apply(lambda i: LABELS[i])
test_file.to_csv('out.csv')
print(f'test score: acc {acc_sum}, score: {score_sum}/{score_all}')
