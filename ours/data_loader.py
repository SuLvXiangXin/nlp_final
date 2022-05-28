import torch
from transformers import BertModel, BertTokenizer
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import random

class MyDataSet(Dataset):
    def __init__(self, csv_file_path):
        self.file = pd.read_csv(csv_file_path)
        # pretrained bert tokenizer
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        self.labels = ['agree', 'disagree', 'discuss', 'unrelated']

    def __len__(self):
        return len(self.file)

    def __getitem__(self, idx):
        head = self.file['Headline'][idx]
        body = self.file['articleBody'][idx]
        stance = torch.tensor(self.labels.index(self.file['Stance'][idx]))
        body_split = body.split(' ')
        if len(body_split) > 512:
            k = random.randint(0, len(body_split)-512)
            body = ' '.join(body_split[k:])
        batch = self.tokenizer(head, body, truncation=True, padding="max_length", max_length=512)
        batch = {k: torch.tensor(v) for k, v in batch.items()}
        batch['stance'] = stance
        return batch


if __name__ == '__main__':
    batch_size = 4
    p = 0.8
    train_dataset = MyDataSet("train_merged.csv")
    test_dataset = MyDataSet("test_merged.csv")
    train_set, val_set = torch.utils.data.random_split(train_dataset, [int(len(train_dataset)*p), len(train_dataset)-int(len(train_dataset)*p)])
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    for i in enumerate(train_loader):
        print(i)
        break


