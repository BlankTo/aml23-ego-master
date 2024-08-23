import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from load_feat_2 import load_features_RGB, scale_features
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
        return x

# Model, Loss, Optimizer
model = HierarchicalModel(input_dim=1024, hidden_dim=256, num_classes=len(set(labels_train)))
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
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, 'extracted_feature_analysis/checkpoints/prova_checkpoint_hier.pth')

train_model(model, X_train, y_train, X_test, y_test)

