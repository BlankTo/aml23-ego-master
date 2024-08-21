import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from load_feat import load_features, scale_features, get_numerical_labels

features, labels = load_features('5_frame', remove_errors= True, ret_value= 'verb')
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

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        h_0 = torch.zeros(1, x.size(0), hidden_dim).to(x.device)  # Initial hidden state
        c_0 = torch.zeros(1, x.size(0), hidden_dim).to(x.device)  # Initial cell state
        out, _ = self.lstm(x, (h_0, c_0))
        out = out[:, -1, :]  # Take the output of the last time step
        out = self.fc(out)
        return out

# Model, Loss, Optimizer
hidden_dim = 256
num_layers = 1
model = LSTMClassifier(input_dim=1024, hidden_dim=hidden_dim, num_layers=num_layers, num_classes=len(set(labels)))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, X_train, y_train, X_test, y_test, epochs=20, save= True):
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
            'epoch': epoch,  # Current epoch
            'model_state_dict': model.state_dict(),  # Model parameters
            'optimizer_state_dict': optimizer.state_dict(),  # Optimizer parameters
            'loss': loss,  # Loss value
        }, 'extracted_feature_analysis/checkpoints/prova_checkpoint_lstm.pth')

train_model(model, X_train, y_train, X_test, y_test)

