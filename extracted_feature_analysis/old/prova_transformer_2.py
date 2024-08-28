import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from extracted_feature_analysis.load_feat import load_features_RGB, scale_features
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from torch.nn import Transformer

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
        x = x[-1, :, :]  # Take the last time step output
        x = self.fc(x)
        return x

# Model, Loss, Optimizer
model = TemporalFusionTransformer(input_dim=1024, num_classes=len(set(labels_train)))
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
        }, 'extracted_feature_analysis/checkpoints/prova_checkpoint_transformer.pth')

train_model(model, X_train, y_train, X_test, y_test)

