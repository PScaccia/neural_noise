#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 17:43:36 2024

@author: paoloscaccia
"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np

# Definizione del decoder lineare
class LinearDecoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearDecoder, self).__init__()
        self.decoder = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.decoder(x)

# Funzione per l'early stopping
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, loss):
        if self.best_loss is None:
            self.best_loss = loss
        elif loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = loss
            self.counter = 0

# Creazione di un dataset di esempio
X = np.random.rand(100, 10).astype(np.float32)  # 100 esempi, 10 feature
y = np.random.rand(100, 1).astype(np.float32)   # 100 etichette

# Suddividere in training e validation set
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# Convertiamo i dati in tensori PyTorch
X_train = torch.tensor(X_train)
y_train = torch.tensor(y_train)
X_val = torch.tensor(X_val)
y_val = torch.tensor(y_val)

# Inizializziamo il modello
input_dim = X_train.shape[1]
output_dim = y_train.shape[1]
model = LinearDecoder(input_dim, output_dim)

# Definiamo la funzione di perdita e l'ottimizzatore
criterion = nn.MSELoss()  # Perdita per regressione (puoi usare CrossEntropy per classificazione)
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Inizializziamo l'early stopping
early_stopping = EarlyStopping(patience=10, min_delta=0.001)

# Numero di epoche
n_epochs = 1000

# Ciclo di allenamento
for epoch in range(n_epochs):
    # Modalità di allenamento
    model.train()
    
    # Azzeriamo i gradienti
    optimizer.zero_grad()

    # Passaggio in avanti (forward)
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    # Backpropagation
    loss.backward()
    optimizer.step()
    
    # Modalità di valutazione
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val)

    print(f'Epoch {epoch+1}/{n_epochs}, Loss: {loss.item()}, Validation Loss: {val_loss.item()}')

    # Controllo early stopping
    early_stopping(val_loss.item())
    if early_stopping.early_stop:
        print("Early stopping")
        break

print("Training completo!")