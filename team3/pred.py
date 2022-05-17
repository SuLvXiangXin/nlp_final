# Copyright 2017 Benjamin Riedel
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# http://www.fakenewschallenge.org/

# Import relevant packages and modules
import argparse

from util import *
import random
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm as tqdm

parser = argparse.ArgumentParser()  # 创建parser

parser.add_argument('-l', action='store_true', default=False, help='whether to load')
args = parser.parse_args()
# Prompt for mode
if args.l:
    mode = 'load'
else:
    mode = 'train'


# Set file names
# ../fnc-1/
file_train_instances = "../fnc-1/train_stances.csv"
file_train_bodies = "../fnc-1/train_bodies.csv"
file_test_instances = "../fnc-1/test_stances_unlabeled.csv"
file_test_bodies = "../fnc-1/test_bodies.csv"
file_predictions = 'out.csv'


# Initialise hyperparameters
r = random.Random()
lim_unigram = 5000
target_size = 4
hidden_size = 100
train_keep_prob = 0.6
l2_alpha = 0.00001
learn_rate = 0.01
clip_ratio = 5
batch_size_train = 500
epochs = 90


# Load data sets
raw_train = FNCData(file_train_instances, file_train_bodies)
raw_test = FNCData(file_test_instances, file_test_bodies)
n_train = len(raw_train.instances)


# Process data sets
train_set, train_stances, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer = \
    pipeline_train(raw_train, raw_test, lim_unigram=lim_unigram)
feature_size = len(train_set[0])
test_set = pipeline_test(raw_test, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer)


# Define train func
def train(indices, model, loss_fn, optimizer):
    model.train()
    for i in range(n_train // batch_size_train):
        # get batch of train data
        batch_indices = indices[i * batch_size_train: (i + 1) * batch_size_train]
        batch_features = [train_set[i] for i in batch_indices]
        batch_stances = [train_stances[i] for i in batch_indices]
        batch_features = torch.tensor(batch_features).type(torch.FloatTensor).to(device)
        batch_stances = torch.tensor(batch_stances).type(torch.LongTensor).to(device)
        # computer prediction error
        pred = model(batch_features)
        loss = loss_fn(pred, batch_stances)
        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# Define model
class NetWork(nn.Module):
    def __init__(self, feature_size, hidden_size, train_keep_prob):
        super(NetWork, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=1-train_keep_prob)
        )
        self.linear = nn.Sequential(
            nn.Linear(hidden_size, target_size),
            nn.Dropout(p=1-train_keep_prob)
        )

    def forward(self, x):
        x = self.hidden(x)
        x = self.linear(x)
        # x = F.softmax(x, dim=1)
        return x


# Create model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NetWork(feature_size, hidden_size, train_keep_prob).to(device)

# Define loss
loss_fn = nn.CrossEntropyLoss()

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate, weight_decay=l2_alpha)

# Train model
if mode == 'train':
    # Perform training
    for epoch in tqdm(range(epochs), unit='epoch'):
        indices = list(range(n_train))
        r.shuffle(indices)
        train(indices, model, loss_fn, optimizer)
    # Save model
    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")

# Load model
elif mode == 'load':
    model.load_state_dict(torch.load("model.pth"))

# Perform predicting
model.eval()
test_set = torch.tensor(test_set).type(torch.FloatTensor).to(device)
test_pred = model(test_set)
test_pred = torch.argmax(test_pred, dim=1).tolist()

# Save predictions
save_predictions(test_pred, file_predictions)

# Concat the result
concat_result(file_test_instances, file_predictions)
