import random
import torch.nn.functional as F
from torch import nn
from collections import namedtuple

struct_transition=namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class Memory:
    def __init__(self,capacity_value):
        self.capacity=capacity_value
        self.memory=[]
        self.index=0

    def savememory(self,state,action,nextstate,reward):
        if len(self.memory)<self.capacity:
            self.memory.append(None)
        self.memory[self.index]=struct_transition(state, action, nextstate, reward)
        self.index+=1
        if self.index>=self.capacity:
            self.index-=self.capacity

    def sample(self,size_of_batch):
        return random.sample(self.memory,size_of_batch)


class Network(nn.Module):
    def __init__(self,number_in,number_middle,number_out):
        super(Network,self).__init__()
        self.fc1 = nn.Linear(number_in,number_middle)
        self.fc2 = nn.Linear(number_middle, number_middle)
        self.fc3 = nn.Linear(number_middle, number_out)

    def forward(self,x):
        input=F.relu(self.fc1(x))
        hidden=F.relu(self.fc2(input))
        output=self.fc3(hidden)
        return output
