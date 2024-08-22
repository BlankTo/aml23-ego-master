import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from load_feat import load_features, scale_features, get_numerical_labels

features, labels = load_features('5_frame', mode= 'train', remove_errors= True, ret_value= 'verb')
labels = np.array(labels)

features = scale_features(features, method= 'standard', ret_scaler= False)
print(features.shape)

labels, num_to_verb, verb_to_num = get_numerical_labels(labels)

features = torch.from_numpy(features.reshape(-1, 5, 1024))
labels = torch.from_numpy(labels[::5]).long()

print(features.shape)
print(labels.shape)

# Split data
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size= 0.2, random_state= 42)

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
        x = torch.cat([x_stream1, x_stream2], dim=1)  # Concatenate features from both streams
        x = self.fc(x)
        return x

# Model, Loss, Optimizer
model = DualStreamNetwork(input_dim=1024, num_classes=len(set(labels)))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, X_train, y_train, X_test, y_test, epochs=100, save= True):
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            outputs = model(X_test)
            _, predicted = torch.max(outputs, 1)
            accuracy = accuracy_score(y_test, predicted)
            print(f'Epoch {epoch+1}, Loss: {loss.item()}, Test Accuracy: {accuracy:.4f}')

    if save:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, 'extracted_feature_analysis/checkpoints/prova_checkpoint_dual.pth')

train_model(model, X_train, y_train, X_test, y_test)

