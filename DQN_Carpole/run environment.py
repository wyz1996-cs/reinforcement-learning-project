import gym
from DQN_Carpole import MyAgent
import torch
import numpy as np

maxepisode=1000
env=gym.make('CartPole-v0')
env._max_episode_steps=500
state_number=env.observation_space.shape[0]
action_number=env.action_space.n
agent=MyAgent.Agent(state_number, action_number)
step=0
consecutive_win=0
steplist=np.zeros(10)
for episode in range(maxepisode):
    observation=env.reset()
    state=torch.unsqueeze(torch.from_numpy(observation).type(torch.FloatTensor),0)
    while True:
        action=agent.epsilon_policy(state,episode)
        next_observation,_,done,_=env.step(action.item())
        step+=1
        if done:
            next_state=None
            steplist=np.hstack((steplist[1:],step))
            if step<300:
                reward=torch.FloatTensor([-1])
                consecutive_win=0
            else:
                reward = torch.FloatTensor([1])
                consecutive_win+=1
        else:
            reward=torch.FloatTensor([0])
            next_state=torch.unsqueeze(torch.from_numpy(next_observation).type(torch.FloatTensor),0)
        agent.memory.savememory(state,action,next_state,reward)
        agent.update()
        state=next_state
        if done:
            print("Episode No:",episode+1," current step:",step,"   With last 10 tries avarage=",steplist.mean())
            step = 0
            break
        if(consecutive_win>10):
            print("10 consecutive win")
            exit(0)
