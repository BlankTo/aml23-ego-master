import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from extracted_feature_analysis.old.load_feat import load_features_RGB, scale_features, get_numerical_labels, convert_to_numerical
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

## loading train

pca_comp = 400
lda_on = True

features_train, labels_train = load_features_RGB('5_frame', split= 'D1', mode= 'train', remove_errors= True, ret_value= 'verb')
labels_train = np.array(labels_train)

features_train_scaled, scaler = scale_features(features_train, method= 'standard', ret_scaler= True)

if pca_comp:
    pca = PCA(n_components= pca_comp)
    features_train_scaled = pca.fit_transform(features_train_scaled)

if lda_on:
    lda = LinearDiscriminantAnalysis(n_components= len(set(labels_train)) - 1)
    features_train_scaled = lda.fit_transform(features_train_scaled, labels_train)

labels_train, num_to_verb, verb_to_num = get_numerical_labels(labels_train)

X_train = torch.from_numpy(features_train)
y_train = torch.from_numpy(labels_train).long()

print(f'X_train: {X_train.shape} - y_train: {y_train.shape}')

## loading test

features_test, labels_test = load_features_RGB('5_frame', split= 'D1', mode= 'test', remove_errors= True, ret_value= 'verb')
labels_test = np.array(labels_test)

features_test_scaled = scaler.transform(features_test)

if pca_comp:
    features_test_scaled = pca.transform(features_test_scaled)

if lda_on:
    features_test_scaled = lda.transform(features_test_scaled)

labels_test = convert_to_numerical(labels_test, verb_to_num)

X_test = torch.from_numpy(features_test)
y_test = torch.from_numpy(labels_test).long()

print(f'X_test: {X_test.shape} - y_test: {y_test.shape}')

##

#features, labels = load_features_RGB('5_frame', split= 'D1', mode= 'train', remove_errors= True, ret_value= 'verb')
#labels = np.array(labels)
#
#features = scale_features(features, method= 'standard', ret_scaler= False)
#print(features.shape)
#
#labels, num_to_verb, verb_to_num = get_numerical_labels(labels)
#
#features = torch.from_numpy(features)
#labels = torch.from_numpy(labels).long()
#
## Split data
#X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size= 0.2, random_state= 42)

##

n_neuron = 34

# Define a simple MLP classifier
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, n_neuron)
        self.fc3 = nn.Linear(n_neuron, num_classes)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc3(x)
        return x

# Model, Loss, Optimizer
model = SimpleMLP(input_dim= 1024, num_classes= len(set(labels_train)))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr= 0.001)

# Training loop
def train_model(model, X_train, y_train, X_test, y_test, epochs= 1000, save= True):
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
        }, 'extracted_feature_analysis/checkpoints/prova_checkpoint_no_temp.pth')

train_model(model, X_train, y_train, X_test, y_test)
