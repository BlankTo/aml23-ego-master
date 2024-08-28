import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from extracted_feature_analysis.load_feat import load_features_RGB, scale_features
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

## loading train

pca_comp = False#400
lda_on = False#True

features_train, labels_train = load_features_RGB('5_frame', split= 'D1', mode= 'train')
labels_train = np.array(labels_train)

features_train_scaled, scaler = scale_features(features_train, method= 'standard', ret_scaler= True)

if pca_comp:
    pca = PCA(n_components= pca_comp)
    features_train_scaled = pca.fit_transform(features_train_scaled)

if lda_on:
    lda = LinearDiscriminantAnalysis(n_components= len(set(labels_train)) - 1)
    features_train_scaled = lda.fit_transform(features_train_scaled, labels_train)

X_train = torch.from_numpy(features_train.reshape(-1, 5, 1024))
y_train = torch.from_numpy(labels_train[::5]).long()

print(f'X_train: {X_train.shape} - y_train: {y_train.shape}')

## loading test

features_test, labels_test = load_features_RGB('5_frame', split= 'D1', mode= 'test')
labels_test = np.array(labels_test)

features_test_scaled = scaler.transform(features_test)

if pca_comp:
    features_test_scaled = pca.transform(features_test_scaled)

if lda_on:
    features_test_scaled = lda.transform(features_test_scaled)

X_test = torch.from_numpy(features_test.reshape(-1, 5, 1024))
y_test = torch.from_numpy(labels_test[::5]).long()

print(f'X_test: {X_test.shape} - y_test: {y_test.shape}')

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
model = AttentionClassifier(input_dim=1024, num_classes=len(set(labels_train)))
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

