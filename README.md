<div align='center'>

## Investigating Q-Learning and DQN variants for Traffic Signal Control problem

</div>

### Abstract 
Traffic Signal Control is an urgent problem that needs to be solved in big cities where traffic jams often occur. This will help people reduce travel time on the road, save fuel in the context of increasing gasoline prices and reduce CO2 emissions into the environment. Reinforcement Learning is one of the popular methods used to solve this problem. In this article, we will experiment with Q-Learning and 4 variants of DQN, observing their results with 6 different metrics. Our goal is to investigate the performance of these algorithms in an environment with 16 intersections. The results we obtained show that while Q-Learning is simple, it is surprisingly effective, which is promising for the application of Q-Learning to the Traffic Signal Control problem in practice.

### Authors:
1. [Tan Ngoc Pham](https://github.com/ngctnnnn)
2. [Dang Vu](https://github.com/twelcone)
3. [Dzung Tri Bui](https://github.com/BTrDung)
4. [An Vo](https://github.com/vokhanhan25)

---

### Table of contents 
1. [Introduction](#1-introduction)
2. [Method](#2-method)
3. [Experiments](#3-experiments)
4. [Results](#4-results)
5. [Conclusions](#5-conclusions)

---
### 1. Introduction 
- The vast improvement in urban traffic nowadays leads to a massive amount of vehicles which results in traffic jams. Traffic congestion will lead to bad consequences such as harm to the environment because when CO2 emissions will increase, time-consuming, and increases fuel costs in the context of the burning energy crisis.     

- Meanwhile, most traffic lights nowadays are still controlled with the pre-defined fixed-time plan without considering real-time. Obviously, the daily flow of vehicles can vary in a complex sense; thus, it is not easy to control with a fixed-time plan. This raises a crucial need for traffic signal optimization to alleviate these problems. Statistics are based on vehicle traffic flow, the needs of citizens traveling on the road, and also the population density so that the researchers can estimate the time allocated for each light signal. 

- To make a system like a human, we need "a brain" to make decisions. Recent advances in Machine Learning, have been encouraging more real-life applications even in the Traffic Signal Control field. The most promising method to handle this type of problem is Reinforcement Learning (RL). Solving a problem using RL needs focusing on two requirements: (1) how to represent the environment; and (2) how to model the correlation between environment and decision. We have proposed many methods for automating traffic lights such as traffic light control using policy-gradient and value-function-based reinforcement learning, traffic signal timing via deep reinforcement learning. 

- The idea behind RL is to interact with the environment, receive feedback from the environment to consider whether one action is good or not given a state, and change the action so as to maximize the final results. Nevertheless, real-life-inspired problems often require a massive observation and action space which makes the traditional RL methods, e.g., dynamic programming (DP) based approaches such as Value Iteration, Policy Iteration, and the usual solutions to use function approximation combined with traditional RL known as *Deep Reinforcement Learning*. And the most frequent DRL methods are Deep Q-Network (DQN) families.

In this work, we will carry out the following tasks:
- Compare the difference of 5 algorithms: Q-Learning, DQN, Double DQN, Dueling DQN, D3QN.
- Use 6 metrics to evaluate the performance of algorithms which are queue length, waiting time, average speed, incoming lanes density, outgoing lanes density, and pressure.
- Use different waiting time as objective function.
- Use SUMO to visualize the problem.

### 2. Method 
#### Q-Learning
- Among model-free reinforcement learning methods, *Q-Learning* is the most popular due to its easy implementation and high performance. The Q-Learning algorithm tries to fill in as many values in the Q-table as possible to help its agents find the best policy to follow. Q-values are updated using the Equation below:

$$
Q_{new}\left( s,a\right) =\left( 1-\alpha \right) ~\underset{\text{old value} }{\underbrace{Q\left( s,a\right) }\rule[-0.05in]{0in}{0.2in} \rule[-0.05in]{0in}{0.2in}\rule[-0.1in]{0in}{0.3in}}+\alpha \overset{\text{ learned value}}{\overbrace{\left(R_{t+1}+\gamma \max_{a^{^{\prime }}}Q\left( s^{\prime },a^{\prime }\right) \right) }} 
$$

in which: $Q(s,a)$ is the q-value on taking action $a$ at state $s$, $s'$ and $a$ is respectively the next state and the next action.

#### Deep Q-Network
- As contemporary problems in real life require too many computational resources and memories for Q-table to handle, \textit{Deep Q-Network} (DQN) \cite{dqn} is proposed as a combined version of traditional Q-learning and neural network. DQN could solve more complex problems with fewer memories and computational hardware while still maintaining efficiency. In the case of DQN, we use a deep neural network to approximate Q-values instead of updating manually via Q-table via Equation below.

$$
Q^\pi(s,a) = \mathcal{R} + \gamma \text{ } \underset{a'}{\max} Q^\pi(s', a')
$$

where: $Q^\pi(s, a)$ is the Q-value predicted by the deep neural network.

- Despite the strength of DQN, it often brings over-optimistic results which would end up with early convergence to (bad) local optima. Later improvements on DQN, i.e., DoubleDQN, DuelingDQN, and D3QN, hence pay much attention to stabilizing DQN's results.

#### Double Deep Q-Network
- *Double Deep Q-Network* (DoubleDQN), which is a deep learning-based improvement from the original DQN and Q-Learning, uses an additional network called $\phi$-network to reduce over-estimations and strengthen performance on many different environments. The formulation of DoubleDQN is demonstrated as in Equation:

$$
    \overset{ {\color{orange}\pi \text{ network}}} {\overbrace{Q^\pi(s,a)}} = \mathcal{R} + \gamma \text{ } \underset{{\color{red} \phi \text{ network}}}{\underbrace{ Q^\phi\left(s', \underset{a'}{\max}Q^\pi(s',a')\right)}}
$$

- Dueling Deep Q-Network (DuelingDQN) is presented to use a dueling architecture which explicitly separates the representation of state values and state-dependent action advantages via two separate streams as demonstrated

<div align='center'>
  <img width="480" alt="Screen Shot 2022-07-10 at 13 21 49" src="https://user-images.githubusercontent.com/67086934/178133830-dc8cadcd-b708-4447-9d5e-045902eb6fa7.png">

</div>

#### Dueling Double Deep Q-Network (D3QN)
- DoubleDQN and DuelingDQN could share their own mechanism to each other and combine into one single deep neural architecture called as *Dueling Double Deep Q-Network* (D3QN)

<div align='center'>
  
<img width="341" alt="Screen Shot 2022-07-10 at 13 22 14" src="https://user-images.githubusercontent.com/67086934/178133836-0cbb3a3d-2cc3-4728-9600-e0a7abfa7200.png">

  </div>
  
### 3. Experiments

- **Queue length**       
The total number of vehicles stopping at intersections.     

- **Waiting time**       
Total waiting time of stopped vehicles at intersections.      

- **Average speed**      
The total average speed of approaching vehicles.

- **Incoming lanes density**            
Total number of vehicles in incoming lanes at the intersections.     

- **Outgoing lanes density**      
The total number of vehicles in outgoing lanes at the intersections.      

- **Pressure**        
Total the differences between total incoming lanes density and total outgoing lanes density.

- **Reward function**            

    - Reward at time $t$ is defined as follows:

        $$
            \mathcal{R}_t = D_t - D_{t + 1}
        $$

        in which, $D_t$ and $D_{t + 1}$ are total waiting time of stopped vehicles at intersections at time $t$ and time $t + 1$, respectively. 

    - In other words, the reward function is defined as how much the total delay (sum of the waiting times of all vehicles) changed in relation to the previous time step. The immediate reward value will be larger when the total waiting time in the next step is shorter than the previous step. In fact, the good policies often have the immediate reward fluctuates around zero, because then the waiting time will not change. However, whether it is good or not should be confirmed by observing other metrics such as average speed, totally stopped,...


### 4. Results

<div align='center'>
  
<img width="1020" alt="Screen Shot 2022-07-10 at 13 26 00" src="https://user-images.githubusercontent.com/67086934/178133927-6d167bad-1ffa-4d0e-8c38-21c175fe2aac.png">
</div>

<div align='center'>
  <img width="1025" alt="Screen Shot 2022-07-10 at 13 26 33" src="https://user-images.githubusercontent.com/67086934/178133938-570698fc-45bc-4584-97ce-7cdff8a0a860.png">
</div>

<div align='center'>
  
<img width="959" alt="Screen Shot 2022-07-10 at 13 26 42" src="https://user-images.githubusercontent.com/67086934/178133944-a7f04fbd-f2b7-47c1-b070-cbaaed65d476.png">
</div>

<div align='center'>
  
<img width="959" alt="Screen Shot 2022-07-10 at 13 27 03" src="https://user-images.githubusercontent.com/67086934/178133951-e3758d41-d902-4ccb-863c-d1a9038f30e9.png">
</div>

<div align='center'>
<img width="1008" alt="Screen Shot 2022-07-10 at 13 27 19" src="https://user-images.githubusercontent.com/67086934/178133958-e330fef3-b591-42d4-84f5-e6282d7b3eb7.png">

</div>

<div align='center'>
  <img width="1016" alt="Screen Shot 2022-07-10 at 13 27 42" src="https://user-images.githubusercontent.com/67086934/178133963-bd674104-737b-41a5-acff-7e8cdd437a23.png">
</div>
  
### 5. Conclusions 
In this report, we compared 5 algorithms on 6 metrics and showed the effectiveness of the training process. The graphs show that the performance of the algorithms depends greatly on the number of cars at a time, or in other words, the environment is not stable. In addition, the results also show that although the Q-Learning algorithm is simple, it gives completely superior results. The advantage of the Q-Learning algorithm is that it is simple, easy to understand, easy to implement, compact, and suitable for problems with a small number of states. These strengths will help Q-Learning be easily applied to solve the TSC problem.

### 6. References
[1] A. J. Miller, “Settings for fixed-cycle traffic signals,”
Journal of the Operational Research Society, vol. 14,
no. 4, pp. 373–386, 1963.     

[2] T. Weisheit and R. Hoyer, “Prediction of switching times
of traffic actuated signal controls using support vector
machines,” in Advanced Microsystems for Automotive
Applications 2014. Springer, 2014, pp. 121–129.      

[3] S. S. Mousavi, M. Schukat, and E. Howley, “Traffic light
control using deep policy-gradient and value-function-
based reinforcement learning,” IET Intelligent Transport
Systems, vol. 11, no. 7, pp. 417–423, 2017.      

[4] L. Li, Y. Lv, and F.-Y. Wang, “Traffic signal timing
via deep reinforcement learning,” IEEE/CAA Journal of
Automatica Sinica, vol. 3, no. 3, pp. 247–254, 2016.       

[5] V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Ve-
ness, M. G. Bellemare, A. Graves, M. Riedmiller, A. K.
Fidjeland, G. Ostrovski et al., “Human-level control
through deep reinforcement learning,” nature, vol. 518,
no. 7540, pp. 529–533, 2015.     

[6] F. Dion, H. Rakha, and Y.-S. Kang, “Comparison of delay
estimates at under-saturated and over-saturated pre-timed
signalized intersections,” Transportation Research Part
B: Methodological, vol. 38, no. 2, pp. 99–122, 2004.      

[7] A. J. Miller, “Settings for fixed-cycle traffic signals,”
Journal of the Operational Research Society, vol. 14,
no. 4, pp. 373–386, 1963.    

[8] F. V. Webster, “Traffic signal settings,” Tech. Rep., 1958.      

[9] S.-B. Cools, C. Gershenson, and B. D’Hooghe, “Self-
organizing traffic lights: A realistic simulation,” in Ad-
vances in applied self-organizing systems. Springer,
2013, pp. 45–55.      

[10] I. Porche and S. Lafortune, “Adaptive look-ahead opti-
mization of traffic signals,” Journal of Intelligent Trans-
portation System, vol. 4, no. 3-4, pp. 209–254, 1999.     

[11] E. M. Ahmed, “Continuous genetic algorithm for traffic
signal control,” in 2018 International Conference on
Computer, Control, Electrical, and Electronics Engineer-
ing (ICCCEEE), 2018, pp. 1–5.      
[12] L. Kuyer, S. Whiteson, B. Bakker, and N. Vlassis, “Mul-
tiagent reinforcement learning for urban traffic control
using coordination graphs,” in Joint European Confer-
ence on Machine Learning and Knowledge Discovery in
Databases. Springer, 2008, pp. 656–671.      

[13] P. Mannion, J. Duggan, and E. Howley, “An experimental
review of reinforcement learning algorithms for adaptive
traffic signal control,” Autonomic road transport support
systems, pp. 47–66, 2016.      

[14] C. J. Watkins and P. Dayan, “Q-learning,” Machine
learning, vol. 8, no. 3, pp. 279–292, 1992.     

[15] H. Van Hasselt, A. Guez, and D. Silver, “Deep reinforce-
ment learning with double q-learning,” in Proceedings of
the AAAI conference on artificial intelligence, vol. 30,
no. 1, 2016.      

[16] Z. Wang, T. Schaul, M. Hessel, H. Hasselt, M. Lanctot,
and N. Freitas, “Dueling network architectures for deep
reinforcement learning,” in International conference on
machine learning. PMLR, 2016, pp. 1995–2003.     

[17] Y. Huang, G. Wei, and Y. Wang, “V-d d3qn: the variant
of double deep q-learning network with dueling architec-
ture,” 07 2018, pp. 9130–9135.        

[18] P. A. Lopez, M. Behrisch, L. Bieker-Walz, J. Erdmann,
Y.-P. Fl ̈otter ̈od, R. Hilbrich, L. L ̈ucken, J. Rummel,
P. Wagner, and E. Wießner, “Microscopic traffic
simulation using sumo,” in The 21st IEEE International
Conference on Intelligent Transportation Systems. IEEE,
2018. [Online]. Available: https://elib.dlr.de/124092/           

[19] A. Raffin, A. Hill, A. Gleave, A. Kanervisto,
M. Ernestus, and N. Dormann, “Stable-baselines3:
Reliable reinforcement learning implementations,”
Journal of Machine Learning Research, vol. 22no. 268, pp. 1–8, 2021. [Online]. Available: http://jmlr.org/papers/v22/20-1364.html      

[20] E. Liang, R. Liaw, R. Nishihara, P. Moritz, R. Fox,
K. Goldberg, J. E. Gonzalez, M. I. Jordan, and I. Stoica,
“RLlib: Abstractions for distributed reinforcement learn-
ing,” in International Conference on Machine Learning
(ICML), 2018.      

[21] L. N. Alegre, “SUMO-RL,”
https://github.com/LucasAlegre/sumo-rl, 2019.     

[22] P. P ́alos and  ́A. Husz ́ak, “Comparison of q-learning based
traffic light control methods and objective functions,” in
2020 International Conference on Software, Telecommu-
nications and Computer Networks (SoftCOM). IEEE,
2020, pp. 1–6.     

[23] F. Klinker, “Exponential moving average versus moving
exponential average,” Mathematische Semesterberichte,
vol. 58, no. 1, pp. 97–107, 2011.

