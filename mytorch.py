import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

class Network_setup(nn.Module):
    def __init__(self,number_in,number_middle,number_out):
        super(Network_setup,self).__init__()
        self.fc1 = nn.Linear(number_in,number_middle)
        self.fc2 = nn.Linear(number_middle, number_middle)
        self.fc3 = nn.Linear(number_middle, number_out)

    def forward(self,x):
        input=F.relu(self.fc1(x))
        hidden=F.relu(self.fc2(input))
        output=self.fc3(hidden)
        return output

def train(epoch):
    model.train()
    for data,target in loader_train:
        optimizer.zero_grad()
        output=model(data)
        loss=loss_function(output,target)
        loss.backward()
        optimizer.step()

def test():
    model.eval()
    correct=0
    with torch.no_grad():
        for data,target in loader_test:
            output=model(data)
            _,pridict=torch.max(output.data,1)
            correct+=pridict.eq(target.data.view_as(pridict)).sum()
    total_number=len(loader_test.dataset)
    print('\ncorrectation rate:',correct/total_number*100)


mnist=fetch_openml('mnist_784',cache='True')
x=mnist.data/255
y=mnist.target

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/7,random_state=0)
x_train=torch.Tensor(x_train)
x_test=torch.Tensor(x_test)
y_train=y_train.astype(np.float)
y_test=y_test.astype(np.float)
y_train=torch.LongTensor(y_train)
y_test=torch.LongTensor(y_test)

dataset_train=TensorDataset(x_train,y_train)
dataset_test=TensorDataset(x_test,y_test)
loader_train=DataLoader(dataset_train,batch_size=64,shuffle=True)
loader_test=DataLoader(dataset_test,batch_size=64,shuffle=False)
model=Network_setup(number_in=28*28*1,number_middle=100,number_out=10)

loss_function=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.01)

test()
for epoch in range(3):
    train(epoch)
test()