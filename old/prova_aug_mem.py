import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from extracted_feature_analysis.old.load_feat import load_features_RGB, scale_features, get_numerical_labels

features, labels = load_features_RGB('5_frame', split= 'D1', mode= 'train', remove_errors= True, ret_value= 'verb')
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
        return out

# Model, Loss, Optimizer
model = MemoryAugmentedNetwork(input_dim=1024, hidden_dim=256, num_classes=len(set(labels)))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, X_train, y_train, X_test, y_test, epochs=50, save= True):
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
        }, 'extracted_feature_analysis/checkpoints/prova_checkpoint_aug_mem.pth')

train_model(model, X_train, y_train, X_test, y_test)

