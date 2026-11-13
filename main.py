import gymnasium as gym
import numpy as np
import time 
import matplotlib.pyplot as plt

env = gym.make("MountainCar-v0")

LEARNING_RATE = 0.1 #anything from 0 to 1
DISCOUNT = 0.5 #how important is future actions compared to current reward
EPISODES = 5000


SHOW_EVERY = 4999



DISCRETE_OS_SIZE = [21] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE


epsilon = 0.5
START_EPSILON_DECAYING = 1 
END_EPSILON_DECAYING = EPISODES // 2

epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

q_table = np.random.uniform(low = -2, high = 0, size = (DISCRETE_OS_SIZE) + [env.action_space.n])

ep_rewards = []
aggregate_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}



def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(int))




for episode in range(EPISODES):
    episode_reward = 0

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



    while not done:

        if np.random.random() > epsilon: 
            action = np.argmax(q_table[discrete_state]) # there are 3 actions, left, rght, and nothing
        else: 
            action = np.random.randint(0,env.action_space.n)

        new_state, reward, terminated, truncated, extraInfo = env.step(action)

        episode_reward += reward
       
        new_discrete_state = get_discrete_state(new_state)
        done = terminated or truncated

       
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )]

            new_q = (1 - LEARNING_RATE) * current_q  + LEARNING_RATE * (reward + DISCOUNT  * max_future_q)
            q_table[discrete_state + (action, )] = new_q

        elif new_state[0] >= env.unwrapped.goal_position:
            q_table[discrete_state + (action, )] = 0 
            

        discrete_state = new_discrete_state

    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING: 
        epsilon -= epsilon_decay_value

    ep_rewards.append(episode_reward)

    if not episode % SHOW_EVERY:
        average_reward = sum(ep_rewards[-SHOW_EVERY:])/len(ep_rewards[-SHOW_EVERY:])
        aggregate_ep_rewards['ep'].append(episode)
        aggregate_ep_rewards['avg'].append(average_reward)
        aggregate_ep_rewards['min'].append(min(ep_rewards[-SHOW_EVERY:]))
        aggregate_ep_rewards['max'].append(max(ep_rewards[-SHOW_EVERY:]))

        print(f"Episode: {episode} avg: {average_reward} min: {min(ep_rewards[-SHOW_EVERY:])} max: {max(ep_rewards[-SHOW_EVERY:])}")



env.close()

plt.plot(aggregate_ep_rewards['ep'],   aggregate_ep_rewards['avg'], label = "avg")
plt.plot(aggregate_ep_rewards['ep'],   aggregate_ep_rewards['min'], label = "min")
plt.plot(aggregate_ep_rewards['ep'],   aggregate_ep_rewards['max'], label = "max")
plt.legend(loc=1)
plt.show()