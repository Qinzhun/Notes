#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as  np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import trange, tqdm


# In[2]:


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[3]:


# Hyper parameters
input_size = 1
hidden_size = 20
num_layers = 1
learning_rate = 0.01


# In[4]:


# Generating random data
def generate_data(num_points, seq_length):
    x = np.random.randint(2, size=(num_points, seq_length, 1))
    y = x.sum(axis=1) % 2
    return x, y


# In[5]:


# Recurrent neural network (many to one)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


# In[6]:


model = RNN(input_size, hidden_size, num_layers).to(device)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# In[7]:


train_X, train_Y = generate_data(10000, 3)
dev_X, dev_Y = generate_data(100, 3)


# In[8]:


train_x = torch.from_numpy(train_X) 
train_y = torch.from_numpy(train_Y)
dev_x = torch.from_numpy(dev_X)
dev_y = torch.from_numpy(dev_Y)
print(train_x.size(), train_y.size())
print(dev_x.size(), dev_y.size())


# In[9]:


train = TensorDataset(train_x, train_y)
dev = TensorDataset(dev_x, dev_y)
train_loader = DataLoader(dataset = train, batch_size = 10, shuffle= True)
dev_loader = DataLoader(dataset = dev, batch_size = 10, shuffle= False)


# In[12]:


# Train the model
for i, (seqs, labels) in enumerate(tqdm(train_loader, desc="Processing")):
    seqs = seqs.to(device, dtype=torch.float)
    labels = labels.squeeze_().to(device, dtype=torch.long)
    
    # Forward
    outputs = model(seqs)
    loss = criterion(outputs, labels)
 
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# In[44]:


with torch.no_grad():
    correct = 0
    total = 0
    for seqs, labels in dev_loader:
        seqs = seqs.to(device, dtype=torch.float)
        labels = labels.squeeze().to(device)
        outputs = model(seqs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
        
print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total)) 


# In[ ]:




