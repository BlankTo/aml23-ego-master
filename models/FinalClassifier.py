import torch
from torch import nn
from torch.nn import Transformer
import itertools
import torch.nn.functional as F

class MLP_single_clip(nn.Module):
    def __init__(self, input_size, hidden_dim, num_classes):
        super(MLP_single_clip, self).__init__()
        self.input_size = input_size
        self.mlp = nn.Sequential(
            nn.Linear(self.input_size, hidden_dim),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        logits = self.mlp(x)
        return logits, {"features": {}}

class MLP_flatten(nn.Module):
    def __init__(self, input_size, temporal_dim, hidden_dim, num_classes):
        super(MLP_flatten, self).__init__()
        self.input_size = input_size * temporal_dim
        self.mlp = nn.Sequential(
            nn.Linear(self.input_size, hidden_dim),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        x = x.reshape(-1, self.input_size)
        logits = self.mlp(x)
        return logits, {"features": {}}

class MLP_avg_pooling(nn.Module):
    def __init__(self, input_shape, temporal_dim, hidden_dim, num_classes):
        super(MLP_avg_pooling, self).__init__()
        self.temporal_dim = temporal_dim
        self.channels = input_shape
        
        # Average pooling layer to reduce temporal dimension
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Linear layers
        self.mlp = nn.Sequential(
            nn.Linear(self.channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # x.shape should be (batch_size, temporal_dim, channels)
        x = x.permute(0, 2, 1)  # Change shape to (batch_size, channels, temporal_dim)
        x = self.avg_pool(x)    # Apply average pooling
        x = x.squeeze(-1)       # Remove the temporal dimension
        x = self.mlp(x)         # Pass through MLP
        return x, {"features": {}}

class MLP_max_pooling(nn.Module):
    def __init__(self, input_shape, temporal_dim, hidden_dim, num_classes):
        super(MLP_max_pooling, self).__init__()
        self.temporal_dim = temporal_dim
        self.channels = input_shape
        
        # Max pooling layer to reduce temporal dimension
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Linear layers
        self.mlp = nn.Sequential(
            nn.Linear(self.channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # x.shape should be (batch_size, temporal_dim, channels)
        x = x.permute(0, 2, 1)  # Change shape to (batch_size, channels, temporal_dim)
        x = self.max_pool(x)    # Apply average pooling
        x = x.squeeze(-1)       # Remove the temporal dimension
        x = self.mlp(x)         # Pass through MLP
        return x, {"features": {}}


class MLP_SimpleTempCov(nn.Module):
    def __init__(self, input_dim, temporal_dim, n_classes):
        super(MLP_SimpleTempCov, self).__init__()
        
        self.temporal_conv = nn.Conv1d(in_channels= input_dim, out_channels= input_dim, kernel_size= temporal_dim)
        
        self.fc = nn.Linear(input_dim, n_classes)
    
    def forward(self, x):
        print(x.shape)
        x = x.transpose(1, 2)
        print(x.shape)
        x = self.temporal_conv(x)
        print(x.shape)
        feat = x.view(x.size(0), -1)
        x = self.fc(feat)
        print(x.shape)
        
        return x, {'features': feat}

    
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
    
    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)  # Initial hidden state
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)  # Initial cell state
        out, _ = self.lstm(x, (h_0, c_0))
        feat = out[:, -1, :]  # Take the output of the last time step
        out = self.fc(feat)
        return out, {"features": feat}
    
class LSTM_emg_base(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LSTM_emg_base, self).__init__()
        self.lstm_1 = nn.LSTM(input_dim, 100, 1, batch_first=True)
        self.lstm_2 = nn.LSTM(100, 50, 1, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(50, num_classes)
    
    def forward(self, x):
        #print(x.shape)

        h_0 = torch.zeros(1, x.size(0), 100).to(x.device)  # Initial hidden state
        c_0 = torch.zeros(1, x.size(0), 100).to(x.device)  # Initial cell state
        x, _ = self.lstm_1(x, (h_0, c_0))
        #print(x.shape)

        h_1 = torch.zeros(1, x.size(0), 50).to(x.device)  # Initial hidden state
        c_1 = torch.zeros(1, x.size(0), 50).to(x.device)  # Initial cell state
        out, _ = self.lstm_2(x, (h_1, c_1))
        #print(out.shape)

        feat = out[:, -1, :]  # Take the output of the last time step
        #print(feat.shape)

        feat = self.dropout(feat)

        out = self.fc(feat)
        #print(out.shape)
        return out, {"features": feat}
    
class LSTM_emg_base_base(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LSTM_emg_base_base, self).__init__()
        self.lstm_1 = nn.LSTM(input_dim, 5, 1, batch_first=True)
        self.lstm_2 = nn.LSTM(5, 50, 1, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(50, num_classes)
    
    def forward(self, x):
        #print(x.shape)

        h_0 = torch.zeros(1, x.size(0), 5).to(x.device)  # Initial hidden state
        c_0 = torch.zeros(1, x.size(0), 5).to(x.device)  # Initial cell state
        x, _ = self.lstm_1(x, (h_0, c_0))
        #print(x.shape)

        h_1 = torch.zeros(1, x.size(0), 50).to(x.device)  # Initial hidden state
        c_1 = torch.zeros(1, x.size(0), 50).to(x.device)  # Initial cell state
        out, _ = self.lstm_2(x, (h_1, c_1))
        #print(out.shape)

        feat = out[:, -1, :]  # Take the output of the last time step
        #print(feat.shape)

        feat = self.dropout(feat)

        out = self.fc(feat)
        #print(out.shape)
        return out, {"features": feat}
    
class LSTM_emg_base_base_2(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LSTM_emg_base_base_2, self).__init__()
        self.lstm = nn.LSTM(input_dim, 50, 1, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(50, num_classes)
    
    def forward(self, x):
        #print(x.shape)

        h = torch.zeros(1, x.size(0), 50).to(x.device)  # Initial hidden state
        c = torch.zeros(1, x.size(0), 50).to(x.device)  # Initial cell state
        out, _ = self.lstm(x, (h, c))
        #print(out.shape)

        feat = out[:, -1, :]  # Take the output of the last time step
        #print(feat.shape)

        feat = self.dropout(feat)

        out = self.fc(feat)
        #print(out.shape)
        return out, {"features": feat}
    
class LSTM_emg(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(LSTM_emg, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
    
    def forward(self, x):
        h = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)  # Initial hidden state
        c = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)  # Initial cell state
        out, _ = self.lstm(x, (h, c))
        feat = out[:, -1, :]  # Take the output of the last time step
        out = self.fc(feat)
        return out, {"features": feat}
    
class TemporalConvNet(nn.Module):
    def __init__(self, input_dim, num_channels, num_classes, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation_size = 2 ** i
            in_channels = input_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=(kernel_size-1)),
                       nn.ReLU(),
                       nn.Dropout(dropout)]
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)  # Change shape to (batch_size, input_dim, temporal_dim)
        x = self.network(x)
        feat = x.mean(dim=2)  # Global average pooling over time
        x = self.fc(x)
        return x, {'features': feat}
    
class TemporalFusionTransformer(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=512, nhead=8, num_encoder_layers=3, num_decoder_layers=3):
        super(TemporalFusionTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.transformer = Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, batch_first=True)
        self.fc = nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        # Assuming x shape is (batch_size, seq_len, input_dim)
        x = self.embedding(x)
        x = x.transpose(0, 1)  # Transformer expects (seq_len, batch_size, d_model)
        x = self.transformer(x, x)  # Here, x is used as both source and target for simplicity
        feat = x[-1, :, :]  # Take the last time step output
        x = self.fc(feat)
        return x, {'features': feat}
    
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super(TemporalBlock, self).__init__()
        padding = (kernel_size - 1) * dilation
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                               padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 
                               padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout2 = nn.Dropout(dropout)
        
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()
        
    def forward(self, x):
        residual = x
        
        x = self.conv1(x)
        x = x[:, :, :-self.conv1.padding[0]]  # Remove extra padding
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = x[:, :, :-self.conv2.padding[0]]  # Remove extra padding
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        if self.downsample is not None:
            residual = self.downsample(residual)
            residual = residual[:, :, -x.size(2):]  # Match the size of x
        
        return self.relu(x + residual)

class TemporalConvNet_2(nn.Module):
    def __init__(self, num_inputs, num_channels, num_classes, kernel_size=2, dropout=0.2):
        super(TemporalConvNet_2, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, dilation=dilation_size, dropout=dropout))
        
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], num_classes)  # Final fully connected layer

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.network(x)
        feat = x.mean(dim=2)  # Global average pooling over the temporal dimension
        x = self.fc(feat)
        return x, {'features': feat}
    
class TRN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, relation_type='pairwise'):
        super(TRN, self).__init__()
        self.channels = input_dim
        self.relation_type = relation_type
        
        # Linear layer to process each frame independently
        self.frame_fc = nn.Linear(self.channels, hidden_dim)
        
        # Relation layer to combine information from multiple frames
        if self.relation_type == 'pairwise':
            self.relation_fc = nn.Linear(2 * hidden_dim, hidden_dim)
        elif self.relation_type == 'triple':
            self.relation_fc = nn.Linear(3 * hidden_dim, hidden_dim)
        else:
            raise ValueError("Unsupported relation type. Use 'pairwise' or 'triple'.")
        
        # Final classification layer
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        batch_size = x.size(0)
        temporal_dim = x.size(1)
        
        # Process each frame independently
        x = self.frame_fc(x)  # Shape: (batch_size, temporal_dim, hidden_dim)
        
        # Compute pairwise or triple relations
        if self.relation_type == 'pairwise':
            relations = []
            for i, j in itertools.combinations(range(temporal_dim), 2):
                relation = torch.cat((x[:, i, :], x[:, j, :]), dim=-1)
                relation = self.relation_fc(relation)
                relations.append(relation)
            relations = torch.stack(relations, dim=1)
        elif self.relation_type == 'triple':
            relations = []
            for i, j, k in itertools.combinations(range(temporal_dim), 3):
                relation = torch.cat((x[:, i, :], x[:, j, :], x[:, k, :]), dim=-1)
                relation = self.relation_fc(relation)
                relations.append(relation)
            relations = torch.stack(relations, dim=1)
        
        # Pool over the computed relations
        relations = relations.mean(dim=1)  # Shape: (batch_size, hidden_dim)
        
        # Final classification
        logits = self.classifier(relations)
        
        return logits, {"features": relations}
    
class CNN_base(nn.Module):
    def __init__(self, num_classes):
        super(CNN_base, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=128, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(256*1*73, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        #print(x.shape)
        feat = x.view(-1, 256*1*73 )
        #print(x.shape)
        x = self.dropout(torch.relu(self.fc1(feat)))
        x = self.fc2(x)
        return x, {"features": feat}
    
class CNN(nn.Module):
    def __init__(self, channels, image_shape, n_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels // 2, (2, 5))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(channels // 2, channels // 4, 5)
        self.fc1 = nn.Linear(1472, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, n_classes)

    def forward(self, x):
        #print(f"1 -> {x.shape}")
        x = self.pool(F.relu(self.conv1(x)))
        #print(f"2 -> {x.shape}")
        x = self.pool(F.relu(self.conv2(x)))
        #print(f"3 -> {x.shape}")
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        #print(f"4 -> {x.shape}")
        x = F.relu(self.fc1(x))
        #print(f"5 -> {x.shape}")
        feat = F.relu(self.fc2(x))
        #print(f"6 -> {feat.shape}")
        x = self.fc3(feat)
        #print(f"7 -> {x.shape}")
        return x, {"features": feat}

class Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        """
        [TODO]: the classifier should be implemented by the students and different variations of it can be tested
        in order to understand which is the most performing one """

    def forward(self, x):
        return self.classifier(x), {}
