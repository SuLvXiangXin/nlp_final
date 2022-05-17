"""
Implements a simple Feed Forward MLP that inherits the Sklearn BaseEstimator.
This model is intended to be used for bagging, since it saves its state
into a random file in order to prevent overwriting files of other runnung
instances of this class. The FeedForwardMLP folder should be cleared after each run
via function delete_saved_graph().
Built on tutorial at https://pythonprogramming.net
"""

# import tensorflow as tf
import numpy as np
import os.path as path
import os
import random
import shutil
from sklearn.base import BaseEstimator
from sklearn.utils import compute_sample_weight
from .model import MLP
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F


class MultiThreadingFeedForwardMLP(BaseEstimator):
    # http://danielhnyk.cz/creating-your-own-estimator-scikit-learn/

    """
    Parameters
    ----------
    batch_size : int (default=200)
        Batch mode:             Number of samples = batch_size
        Mini-batch mode:        1 < batch_size < Number of samples
        Stochastic mode:        batch_size = 1
        Paper on this subject:  https://arxiv.org/abs/1609.04836
    """

    def __init__(self, in_channels, n_classes=4, batch_size=200, hm_epochs=15, learning_rate=0.001,
                 hidden_layers=(600, 600, 600), seed=12345, name=None):
        self.in_channels = in_channels
        self.learning_rate = learning_rate
        self.name = name
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.hm_epochs = hm_epochs
        self._estimator_type = "classifier"
        self.hidden_layers = hidden_layers
        self.seed = seed
        self.device = 'cuda'
        self.model = MLP(in_channels, hidden_layers=hidden_layers).to(self.device)
        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = MultiStepLR(self.optimizer, milestones=[20, 35, 45], gamma=0.1)
        self.save_path = 'mlp_models/%s.pth' % name

    def fit(self, X_train, y_train):
        # if os.path.exists(self.save_path):
        #     self.model.load_state_dict(torch.load(self.save_path))
        #     return self
        n = X_train.shape[0]
        for epoch in range(self.hm_epochs):
            epoch_loss = 0
            epoch_acc = 0
            # increase momentum steadily
            i = 0
            while i < len(X_train):
                start = i
                end = i + self.batch_size
                batch_x = torch.from_numpy(X_train[start:end]).float().to(self.device)
                batch_y = torch.from_numpy(y_train[start:end]).to(self.device)
                self.optimizer.zero_grad()
                batch_pred = self.model(batch_x)
                batch_label = batch_pred.argmax(1)
                acc = (batch_label == batch_y).sum()
                loss = self.loss_func(batch_pred, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                epoch_acc += acc.item()
                i += self.batch_size

            self.scheduler.step()
            print('Epoch', epoch + 1, '/', self.hm_epochs, 'loss:', epoch_loss / n, 'acc:', epoch_acc / n)
        torch.save(self.model.state_dict(), self.save_path)

        return self

    def predict(self, X_test):
        i = 0
        labels = []
        with torch.no_grad():
            while i < len(X_test):
                start = i
                end = i + self.batch_size
                batch_x = torch.from_numpy(X_test[start:end]).float().to(self.device)
                batch_pred = self.model(batch_x)
                batch_label = batch_pred.argmax(1)
                labels.append(batch_label.detach().cpu().numpy())
                i += self.batch_size
        return np.concatenate(labels, axis=0)

    def predict_proba(self, X_test):
        i = 0
        probs = []
        with torch.no_grad():
            while i < len(X_test):
                start = i
                end = i + self.batch_size
                batch_x = torch.from_numpy(X_test[start:end]).to(self.device)
                prob = F.softmax(self.model(batch_x)).detach().cpu().numpy()
                probs.append(prob)
        return np.concatenate(prob, axis=0)
