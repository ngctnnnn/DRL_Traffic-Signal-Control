<div align='center'>

## Investigating Q-Learning and DQN variants for Traffic Signal Control problem

</div>

### Abstract 
Traffic Signal Control is an urgent problem that needs to be solved in big cities where traffic jams often occur. This will help people reduce travel time on the road, save fuel in the context of increasing gasoline prices and reduce CO2 emissions into the environment. Reinforcement Learning is one of the popular methods used to solve this problem. In this article, we will experiment with Q-Learning and 4 variants of DQN, observing their results with 6 different metrics. Our goal is to investigate the performance of these algorithms in an environment with 16 intersections. The results we obtained show that while Q-Learning is simple, it is surprisingly effective, which is promising for the application of Q-Learning to the Traffic Signal Control problem in practice.

---

### Table of contents 


---
#### 1. Introduction 
The vast improvement in urban traffic nowadays leads to a massive amount of vehicles which results in traffic jams. Traffic congestion will lead to bad consequences such as harm to the environment because when CO2 emissions will increase, time-consuming, and increases fuel costs in the context of the burning energy crisis.     

Meanwhile, most traffic lights nowadays are still controlled with the pre-defined fixed-time plan without considering real-time. Obviously, the daily flow of vehicles can vary in a complex sense; thus, it is not easy to control with a fixed-time plan. This raises a crucial need for traffic signal optimization to alleviate these problems. Statistics are based on vehicle traffic flow, the needs of citizens traveling on the road, and also the population density so that the researchers can estimate the time allocated for each light signal. 

To make a system like a human, we need "a brain" to make decisions. Recent advances in Machine Learning, have been encouraging more real-life applications even in the Traffic Signal Control field. The most promising method to handle this type of problem is Reinforcement Learning (RL). Solving a problem using RL needs focusing on two requirements: (1) how to represent the environment; and (2) how to model the correlation between environment and decision. We have proposed many methods for automating traffic lights such as traffic light control using policy-gradient and value-function-based reinforcement learning, traffic signal timing via deep reinforcement learning. 

The idea behind RL is to interact with the environment, receive feedback from the environment to consider whether one action is good or not given a state, and change the action so as to maximize the final results. Nevertheless, real-life-inspired problems often require a massive observation and action space which makes the traditional RL methods, e.g., dynamic programming (DP) based approaches such as Value Iteration, Policy Iteration, and the usual solutions to use function approximation combined with traditional RL known as *Deep Reinforcement Learning*. And the most frequent DRL methods are Deep Q-Network (DQN) families.

In this work, we will carry out the following tasks:
- Compare the difference of 5 algorithms: Q-Learning, DQN, Double DQN, Dueling DQN, D3QN.
- Use 6 metrics to evaluate the performance of algorithms which are queue length, waiting time, average speed, incoming lanes density, outgoing lanes density, and pressure.
- Use different waiting time as objective function.
- Use SUMO to visualize the problem.

#### 2. Method 
##### Q-Learning
Among model-free reinforcement learning methods, *Q-Learning* is the most popular due to its easy implementation and high performance. The Q-Learning algorithm tries to fill in as many values in the Q-table as possible to help its agents find the best policy to follow. Q-values are updated using the Equation below, in which $Q(s,a)$ is the q-value on taking action $a$ at state $s$, $s'$ and $a'$ is respectively the next state and the next action, and $\gamma, \gamma \in \left[0;1\right]$ is the discounted factor.

$$
Q_{new}\left( s,a\right) =\left( 1-\alpha \right) ~\underset{\text{old value} }{\underbrace{Q\left( s,a\right) }\rule[-0.05in]{0in}{0.2in} \rule[-0.05in]{0in}{0.2in}\rule[-0.1in]{0in}{0.3in}}+\alpha \overset{\text{ learned value}}{\overbrace{\left(R_{t+1}+\gamma \max_{a^{^{\prime }}}Q\left( s^{\prime },a^{\prime }\right) \right) }} 
$$

#### 3. Experiments

#### 4. Results

#### 5. Conclusions 

#### 6. References
