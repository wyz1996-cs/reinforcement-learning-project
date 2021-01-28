from DQN_Carpole import Networkparts
import random
import torch
import torch.nn.functional as F
from torch import optim

batchsize=32
gamma=0.99

class Agent:
    def __init__(self,number_of_state,number_of_action):
        self.action_number=number_of_action
        self.memory=Networkparts.Memory(10000)
        self.network=Networkparts.Network(number_of_state,batchsize,number_of_action)
        self.optimizer=optim.Adam(self.network.parameters(),lr=0.0001)
    def update(self):
        if len(self.memory.memory)<batchsize:
            return
        experience_sample=self.memory.sample(batchsize)
        liststate, listaction, listnextstate, listreward = [], [], [], []
        for i in range(len(experience_sample)):
            liststate.append(experience_sample[i].state)
            listaction.append(experience_sample[i].action)
            listnextstate.append(experience_sample[i].next_state)
            listreward.append(experience_sample[i].reward)
        liststate=torch.cat(liststate)
        listaction=torch.cat(listaction)
        listreward=torch.cat(listreward)
        next_state=torch.cat([s for s in listnextstate if s is not None])



        self.network.eval()
        actual_qvalue=self.network(liststate).gather(1,listaction)
        index=torch.ByteTensor(batchsize)
        for i in range(batchsize):
            if listnextstate[i]==None:
                index[i]=0
            else:
                index[i]=1
        maxQ=torch.zeros(batchsize)
        maxQ[index]=self.network(next_state).max(1)[0].detach()
        target_qvalue=listreward+gamma*maxQ
        self.network.train()
        loss=F.smooth_l1_loss(actual_qvalue,target_qvalue.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def epsilon_policy(self,state,episode):
        epsilon=0.5*(1/(episode+1))
        if epsilon<random.uniform(0,1):
            self.network.eval()
            with torch.no_grad():
                action=self.network(state).max(1)[1].view(1,1)
        else:
            action =torch.LongTensor([[random.randrange(self.action_number)]])
        return action