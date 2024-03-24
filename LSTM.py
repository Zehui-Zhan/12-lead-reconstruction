import argparse
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch
import glob
import matplotlib.pyplot as plt
import scipy.io
import random
import torch
import torch.nn as nn
class Generator_lstm(nn.Module):
    def __init__(self, length=512,in_channel=1,out_channel=11):
        super(Generator_lstm, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channel, out_channels=128, kernel_size=5,stride=1,padding=2)
        self.maxpool = nn.MaxPool1d(2,stride=2)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5,stride=1,padding=2)
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=1)
        self.dense = nn.Linear(8192,1024)
        self.drop = nn.Dropout(0.2)
        self.conv3 = nn.Conv1d(in_channels=1, out_channels=out_channel, kernel_size=5,stride=1,padding=2)

    def forward(self,z):
        x1 = self.conv1(z)
        x2 = self.maxpool(x1)
        x3 = self.conv2(x2)
        x4 = self.maxpool(x3)
        x5 = self.conv2(x4)
        x6 = self.maxpool(x5)
        x7 = x6.permute(0,2,1)
        x8 = self.lstm(x7)
        x8 = x8[0]
        x8 = torch.flatten(x8,start_dim=1)
        x8 = x8.unsqueeze(1)
        x8 = self.drop(x8)
        x9 = self.dense(x8)
        x10 = self.conv3(x9)
        return x10

def loss_function(y_hat,y):
    loss = F.mse_loss(y_hat, y)
    return loss

