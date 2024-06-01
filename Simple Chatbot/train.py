import json
import numpy as np
from nltk_utils import tokenize, stem, bag_of_words
from model import NeuralNetwork
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


with open('intents.json', 'r') as f:
    intents=json.load(f)
# print(intents)
all_words=[]
tags=[]
xy=[]
for intent in intents['intents']: # key is 'intents'
    tag=intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w=tokenize(pattern)
        all_words.extend(w) # w is array, so extend is used instead of append
        xy.append((w, tag))

ignore_words=['?','!','.',',']
all_words=[stem(w) for w in all_words if w not in ignore_words] # stem + lowercase + remove punctuation
all_words=sorted(set(all_words)) # set removes duplicate
# tags=sorted(set(tags)) # must not sort as tags is used later to get index
print(all_words)
print(tags)

x_train=[]
y_train=[]
for (pattern,tag) in xy: # as pattern and tag are inside tuple
    bag=bag_of_words(pattern, all_words)
    x_train.append(bag)

    label=tags.index(tag) # sorted tags may give faulty index compared to tag
    y_train.append(label)

print(x_train[20], y_train[20])
print(type(y_train[0]))
x_train=np.array(x_train)
y_train=np.array(y_train)
y_train = torch.from_numpy(y_train).long()  # Convert to LongTensor
'''
Must convert int to scalar
model expects long, not int
'''
print(type(y_train[0]))

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples=len(x_train)
        self.x_data=x_train
        self.y_data=y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.n_samples
    
# Hyperparameters
batch_size=2
hidden_size=32
input_size=len(x_train[0])
num_classes=len(tags)
print(input_size, len(all_words))
learning_rate=0.00109
num_epochs=811 # total epochs 19+91+118+811=1039

dataset=ChatDataset()
train_loader=DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True) # may use num_workers for multi thread

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model=NeuralNetwork(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words=words.to(device)
        labels=labels.to(device)

        # forward
        outputs = model(words)
        loss = criterion(outputs, labels)

        # backward and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch: {epoch+1} Loss: {loss.item():.16f}')

data={
    'model_state':model.state_dict(),
    'input_size':input_size,
    'num_classes':num_classes,
    'hidden_size':hidden_size,
    'all_words':all_words,
    'tags':tags
}

FILE = 'data.pth' # saved both pt and pth file for more convenience
torch.save(data,FILE)