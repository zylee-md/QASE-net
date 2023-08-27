import torch
import torch.nn as nn

class CNN1d(nn.Module):
    def __init__(self):
        super(CNN1d, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=16, stride=8),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=8, stride=4),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=2, stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )
        
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = torch.mean(x, dim=2)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class CNNBLSTM(nn.Module):
    def __init__(self):
        super(CNNBLSTM, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=16, stride=8),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=8, stride=4),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=128, kernel_size=4, stride=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, bidirectional=True, batch_first=True)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = torch.mean(x, dim=1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x
    
class CNNATTN(nn.Module):
    def __init__(self):
        super(CNNATTN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=16, stride=8),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=8, stride=4),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )
        self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=2, batch_first=True)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.permute(0, 2, 1)
        x, _ = self.attention(x, x, x)
        x = torch.mean(x, dim=1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x    
    
class CNNBLSTMATTN(nn.Module):
    def __init__(self):
        super(CNNBLSTMATTN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=16, stride=8),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=8, stride=4),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True)
        )
        self.lstm = nn.LSTM(input_size=32, hidden_size=64, num_layers=1, bidirectional=True, batch_first=True)
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        self.fc1 = nn.Linear(128, 64)  
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x, _ = self.attention(x, x, x)
        x = torch.mean(x, dim=1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

class WLDNN(nn.Module):
    def __init__(self, input_length):
        super(WLDNN, self).__init__()
        self.hidden1 = nn.Linear(input_length, 20)
        self.hidden2 = nn.Linear(20, 20)
        self.output = nn.Linear(20, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.hidden1(x)
        x = self.sigmoid(x)
        x = self.hidden2(x)
        x = self.sigmoid(x)
        x = self.output(x)
        return x