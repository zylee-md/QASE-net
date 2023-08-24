import torch
import torch.nn as nn
import torch.nn.functional as F

# Convolutional Neural Network for 1D data
class CNN1d(nn.Module):
    def __init__(self):
        super(CNN1d, self).__init__()
        # Conv layers
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
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=8, stride=4),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=4, stride=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=4, stride=2),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True)
        )

        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # Pool to 1 feature along the time axis
        
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(256, 128)  # Adjust input size
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        # Apply average pooling along the time axis
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Convolutional Neural Network with Bidirectional LSTM
class CNNLSTM(nn.Module):
    def __init__(self):
        super(CNNLSTM, self).__init__()
        # Conv layers
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
        
        # Bidirectional LSTM layers
        self.lstm = nn.LSTM(input_size=32, hidden_size=256, num_layers=2, bidirectional=True, batch_first=True)

        # Linear layers
        self.fc1 = nn.Linear(512, 128)  # Input size: 512 (bidirectional LSTM output)
        self.fc2 = nn.Linear(128, 1)    # Output a single number

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        
        # Reshape the output to match LSTM input size
        x = x.permute(0, 2, 1)  # Change channel dimension to match LSTM input
        lstm_out, _ = self.lstm(x)
        
        # Global average pooling along the time axis
        x = torch.mean(lstm_out, dim=1)

        # Linear layers
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)

        return x

class CNNAttention(nn.Module):
    def __init__(self, num_attention_heads=4):
        super(CNNAttention, self).__init__()
        
        # Conv layers
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
        
        # Multi-Head Self-Attention Layer
        self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=num_attention_heads)
        
        # Calculate the input dimension for the first linear layer
        input_dim = self.calculate_input_dimension(10000)
        
        # Linear layers
        self.fc1 = nn.Linear(input_dim, 128)  # Input size calculated based on input signal length
        self.fc2 = nn.Linear(128, 1)  # Output a single number

    def calculate_input_dimension(self, input_length):
        # Calculate the output sequence length after the convolutional layers
        conv1_output_length = ((input_length - 16) // 8) + 1
        conv2_output_length = ((conv1_output_length - 8) // 4) + 1
        conv3_output_length = ((conv2_output_length - 4) // 2) + 1
        
        # Calculate the input dimension for the first linear layer
        input_dim = 64 * conv3_output_length
        
        return input_dim

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Reshape the output for Multi-Head Self-Attention
        x = x.permute(2, 0, 1)  # Change channel dimension to match Multi-Head Self-Attention input
        
        # Multi-Head Self-Attention Layer
        x, _ = self.attention(x, x, x)
        
        # Flatten
        x = x.view(x.size(1), -1)  # Reshape for fully connected layers
        
        # Linear layers
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)

        return x


# Deep Neural Network
class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.linear1 = nn.Linear(10000, 256)
        self.linear2 = nn.Linear(256, 256)
        self.output = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.output(x)
        return x

# Deep Neural Network that takes WL as input
class WLDNN(nn.Module):
    def __init__(self, input_length):
        super(WLDNN, self).__init__()
        self.hidden1 = nn.Linear(input_length, 20)
        self.hidden2 = nn.Linear(20, 20)
        self.output = nn.Linear(20, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.sigmoid(self.hidden1(x))
        x = self.sigmoid(self.hidden2(x))
        x = self.output(x)
        return x