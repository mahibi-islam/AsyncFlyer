import torch
from torch import nn

class FLyerLSTM(nn.Module):
    def __init__(self, input_size: int, num_classes: int, lstm_units: int = 64, dense_units: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=lstm_units, batch_first=True)
        self.dense1 = nn.Linear(lstm_units, dense_units)
        self.dense2 = nn.Linear(dense_units, dense_units)
        self.out = nn.Linear(dense_units, num_classes)
        self.relu = nn.ReLU()
    def forward(self, x):
        sequence, _ = self.lstm(x)
        h = sequence[:, -1, :]
        h = self.relu(self.dense1(h))
        h = self.relu(self.dense2(h))
        return self.out(h)
