import torch
from torch import nn
from torch.nn import Transformer

class MLP(nn.Module):
    def __init__(self, num_classes, batch_size):
        super(MLP, self).__init__()
        self.input_size = 1024
        self.mlp = nn.Sequential(
            nn.Linear(self.input_size, 512),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        logits = self.mlp(x)
        return logits, {"features": {}}
    
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
    
class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.W = nn.Linear(input_dim, input_dim)
        self.v = nn.Linear(input_dim, 1, bias=False)
    
    def forward(self, x):
        scores = self.v(torch.tanh(self.W(x)))
        weights = torch.softmax(scores, dim=1)
        weighted_sum = torch.sum(weights * x, dim=1)
        return weighted_sum

class AttentionClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(AttentionClassifier, self).__init__()
        self.attention = AttentionLayer(input_dim)
        self.fc = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        feat = self.attention(x)
        x = self.fc(feat)
        return x, {'features': feat}
    
class MemoryAugmentedNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(MemoryAugmentedNetwork, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.memory = nn.Linear(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        # x shape: (batch_size, clip_in_sample, features_per_clip)
        out, _ = self.lstm(x)
        memory = self.memory(out[:, -1, :])
        out = self.fc(memory)
        return out, {'features': memory}
    
class CombinedModel(nn.Module):
    def __init__(self, input_dim, mlp_dim, hidden_dim, num_layers, num_classes):
        super(CombinedModel, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, mlp_dim),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(mlp_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.hidden_dim = hidden_dim
    
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        x = x.view(-1, x.size(-1))  # Reshape to (batch_size*seq_len, input_dim)
        x = self.mlp(x)
        x = x.view(batch_size, seq_len, -1)  # Reshape back to (batch_size, seq_len, mlp_dim)
        h_0 = torch.zeros(1, x.size(0), self.hidden_dim).to(x.device)
        c_0 = torch.zeros(1, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h_0, c_0))
        feat = out[:, -1, :]
        out = self.fc(feat)
        return out, {'features': feat}
    
class DualStreamNetwork(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(DualStreamNetwork, self).__init__()
        self.stream1 = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        self.stream2 = nn.LSTM(input_dim, 256, batch_first=True)
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        # x shape: (batch_size, clip_in_sample, features_per_clip)
        x_stream1 = self.stream1(x.mean(dim=1))  # Static features
        x_stream2, _ = self.stream2(x)  # Temporal features
        x_stream2 = x_stream2[:, -1, :]  # Take the last time step output
        feat = torch.cat([x_stream1, x_stream2], dim=1)  # Concatenate features from both streams
        x = self.fc(feat)
        return x, {'feautures': feat}
    
class HierarchicalModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(HierarchicalModel, self).__init__()
        self.lstm_clip = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.lstm_video = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        # x shape: (batch_size, clip_in_sample, features_per_clip)
        clip_features, _ = self.lstm_clip(x)  # Process each clip
        video_features, _ = self.lstm_video(clip_features)  # Process sequence of clips
        video_features = video_features[:, -1, :]  # Take the output of the last time step
        x = self.fc(video_features)
        return x, {'features': video_features}
    
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

class Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        """
        [TODO]: the classifier should be implemented by the students and different variations of it can be tested
        in order to understand which is the most performing one """

    def forward(self, x):
        return self.classifier(x), {}
