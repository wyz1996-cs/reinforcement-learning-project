import random

import numpy as np
import gym


def bins(low,high,levels):
    return np.linspace(low,high,levels+1)[1:-1]

def normalise_state(observation):
    car_position,car_velocity,pole_angle,pole_velocity=observation
    digitize=[
        np.digitize(car_position,bins(-2.4,2.4,6)),
        np.digitize(car_velocity, bins(-3, 3, 6)),
        np.digitize(pole_angle, bins(-0.5, 0.5, 6)),
        np.digitize(pole_velocity, bins(-2, 2, 6))
    ]

    return sum([x*(6**i) for i,x in enumerate(digitize)])



env=gym.make('CartPole-v0')

observation_number=env.observation_space.shape[0]
action_number=env.action_space.n
q_table = np.random.uniform(low=0,high=1,size=(6**observation_number,action_number))




episodes = 1000
alpha = 0.5
gamma = 0.7
step_total=0


for episode in range(episodes):
    observation=env.reset()
    #observation,_,_,_=env.step(env.action_space.sample())
    step=0
    current_state=normalise_state(observation)
    epsilon = 0.5*(1/(episode+1))
    while True:
        # for half change, it go through policy, another half take random action
        if random.random() > epsilon:
            next_actions = np.argmax(q_table[current_state][:])
        else:
            next_actions = random.randint(0, 1)
        next_observation,_,done,_=env.step(next_actions)
        step+=1
        next_state=normalise_state(next_observation)
        if done:
            if step<200:
                reward=-1

            else:
                reward = 1

        else:
            reward = 0
        # update q table and policy

        q_table[current_state,next_actions] += alpha * (reward + gamma * max(q_table[next_state][:]) - q_table[current_state,next_actions])
        current_state = next_state

        # if reward is 100, current episode ends
        if done:
            step_total+=step
            if episode % 100 == 99:
                avg = step_total / 100
                step_total=0
                print(avg, "  ", episode+1)
            break


