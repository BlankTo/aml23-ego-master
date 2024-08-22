import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from load_feat import load_features, scale_features, get_numerical_labels

from torch.nn import Transformer

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

class TemporalFusionTransformer(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=512, nhead=8, num_encoder_layers=3, num_decoder_layers=3):
        super(TemporalFusionTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.transformer = Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers)
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
model = TemporalFusionTransformer(input_dim=1024, num_classes=len(set(labels)))
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

