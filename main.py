import gymnasium as gym
import numpy as np
import time 

env = gym.make("MountainCar-v0")

LEARNING_RATE = 0.1 #anything from 0 to 1
DISCOUNT = 0.95 #how important is future actions compared to current reward
EPISODES = 25000


SHOW_EVERY = 500



DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE


epsilon = 0 
START_EPSILON_DECAYING = 1 
END_EPSILON_DECAYING = EPISODES // 2

epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

q_table = np.random.uniform(low = -2, high = 0, size = (DISCRETE_OS_SIZE) + [env.action_space.n])


def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(int))




for episode in range(EPISODES):

    if (episode%SHOW_EVERY == 0):
        render = True
        print(episode)
    else:
        render = False 


    env.close()
    env = gym.make(
        "MountainCar-v0",
        render_mode = "human" if render else None
    )

    discrete_state = get_discrete_state(env.reset()[0])
    done = False

    stepAmount = 0


    while not done:

        if np.random.random() > epsilon: 
            action = np.argmax(q_table[discrete_state]) # there are 3 actions, left, rght, and nothing
        else: 
            action = np.random.randint(0,env.action_space.n)

        new_state, reward, terminated, truncated, extraInfo = env.step(action)

        stepAmount += 1
       
        new_discrete_state = get_discrete_state(new_state)
        done = terminated or truncated

       
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )]

            new_q = (1 - LEARNING_RATE) * current_q  + LEARNING_RATE * (reward + DISCOUNT  * max_future_q)
            q_table[discrete_state + (action, )] = new_q

        elif new_state[0] >= env.unwrapped.goal_position:
            q_table[discrete_state + (action, )] = 0 
            print(episode)
            

        discrete_state = new_discrete_state

    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING: 
        epsilon -= epsilon_decay_value




env.close()
