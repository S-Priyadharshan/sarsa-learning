# SARSA Learning Algorithm


## AIM
The aim of the experiment is the demonstrate the SARSA learning algorithm and compare it to the the Monte Carlo algorithm as well as the optimal policy to understand about its performance. 

## PROBLEM STATEMENT

The problem involves simulating the Temporal difference algorithm on a given MDP (Frozen Lake for this experiment) and understand how it evaluates a given environment as compared to other types of RL algorithms such as value iteration or Monte Carlo prediction/Control algorithm and the various tradeoffs and advantages that it presents to us. We make use of the SARSA (state-action-reward-state-action) function to help us carry out this task.

## SARSA LEARNING ALGORITHM

### STEP 1: 
We take a frozen Lake MDP to simulate our comparisons.

### STEP 2: 
Obtain the optimal policy,value function and action function through the process of value iteration and print its output as a base reference.

### STEP 3: 
Next we run the MC control algorithm on the environment and simulate the episode-by-episode process and derivate it's policy,value and action functions.

### STEP 4: 
Finally we run the Temporal differencing based SARSA algorithm and obtain the policy,value and action functions based on its step-by-step updation procedure.

### STEP 5: 
We then compare all three to determine which algorithm is suitable for which type of environment and the different constraints that they solve.

## SARSA LEARNING FUNCTION
### Name: Priyadharshan S
### Register Number: 212223240127

```python
def sarsa(env,
          gamma=1.0,
          init_alpha=0.5,
          min_alpha=0.01,
          alpha_decay_ratio=0.5,
          init_epsilon=1.0,
          min_epsilon=0.1,
          epsilon_decay_ratio=0.9,
          n_episodes=3000):
    nS, nA = env.observation_space.n, env.action_space.n
    pi_track = []
    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)

    select_action = lambda state,Q,epsilon: np.argmax(Q[state]) if np.random.random() > epsilon else np.random.randint(len(Q[state]))

    alphas = decay_schedule(init_alpha,
                            min_alpha,
                            alpha_decay_ratio,
                            n_episodes)
    
    epsilons = decay_schedule(init_epsilon,
                              min_epsilon,
                              epsilon_decay_ratio,
                              n_episodes)
    
    for e in tqdm(range(n_episodes), leave=False):

      state,done = env.reset(),False;
      action = select_action(state,Q,epsilons[e])

      while not done:

        next_s,reward,done,_ = env.step(action);
        next_a = select_action(next_s,Q,epsilons[e])

        td_target = reward + gamma * Q[next_s][next_a] * (not done)

        td_error = td_target - Q[state][action]

        Q[state][action] = Q[state][action]+alphas[e]*td_error

        state,action = next_s,next_a

      Q_track[e] = Q
      pi_track.append(np.argmax(Q,axis=1))
      
    V = np.max(Q,axis=1)
    pi = lambda s: {s:a for s,a in enumerate(np.argmax(Q,axis=1))}[s]
    return Q, V, pi, Q_track, pi_track
```

## OUTPUT:
<img width="2485" height="777" alt="image" src="https://github.com/user-attachments/assets/95edaf46-9265-474a-9570-7396e7995013" />

<img width="2472" height="777" alt="image" src="https://github.com/user-attachments/assets/5b2acb16-16db-4fd8-bf99-9c8ba766a8fd" />

## RESULT:

Thus we have successfully simulated the SARSA algorithm and compared its output.
