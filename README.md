# reinforcement-learning-project
grid.py used Q-learning to find path in a 12*12 grid world.

carpole:fixed version. I think the problem is because mistakes in digitize phase. The reward is set +1 when step>200, so it will converge to 200 in the end.

mytorch: try pytorch and used it to do mnist classify task, 95% accuracy as result 

DQN Carpole: Network_part contains two function 1)network, the network used to train to get q-value 2) memory, the experience pool used to save data for trainning

Agent: implement update logic for network and q learning
       policy function decide next action
       
run environemnt: set up the whole iterative env and execute, the goal is to reach 300 step on cartpole in 10 consecutive runs. Result is ~150 episode
