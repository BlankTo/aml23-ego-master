import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from load_feat import load_features, scale_features, get_numerical_labels

features, labels = load_features('5_frame', split= 'D1', mode= 'train', remove_errors= True, ret_value= 'verb')
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
        x = self.attention(x)
        x = self.fc(x)
        return x

# Model, Loss, Optimizer
model = AttentionClassifier(input_dim=1024, num_classes=len(set(labels)))
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
        }, 'extracted_feature_analysis/checkpoints/prova_checkpoint_attention.pth')

train_model(model, X_train, y_train, X_test, y_test)

