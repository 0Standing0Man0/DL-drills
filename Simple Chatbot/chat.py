import random
import json
import torch
from model import NeuralNetwork
from nltk_utils import tokenize, bag_of_words

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)

with open('intents.json', 'r') as f:
    intents=json.load(f)

FILE='data.pt'
data=torch.load(FILE)
input_size=data['input_size']
hidden_size=data['hidden_size']
num_classes=data['num_classes']
all_words=data['all_words']
tags=data['tags']
model_state=data['model_state']

model=NeuralNetwork(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name='Chat Buddy'
print('W E L C O M E !')
print('type "end chat" to end chat')
while True:

    sentence=input('Human: ')

    if sentence == 'end chat':
        break

    sentence=tokenize(sentence=sentence)
    x=bag_of_words(sentence,all_words=all_words)
    x=x.reshape(1,x.shape[0]) # 1 row as only 1 sample
    x=torch.from_numpy(x)

    output=model(x)
    _, predicted=torch.max(output, dim=1)
    tag=tags[predicted.item()]

    probs=torch.softmax(output, dim=1)
    prob=probs[0][predicted.item()]

    if prob >= 0.75:
        for intent in intents['intents']:
            if tag==intent['tag']:
                print(f'{bot_name}: {random.choice(intent["responses"])}')
    else:
        print(f'{bot_name}: SPEAK ENGLISH!')