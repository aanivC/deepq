import gymnasium as gym
import matplotlib.pyplot as mpl
import numpy as np 
import math
import time 
import random



env = gym.make("Acrobot-v1")

""""
we want our variables here
Q(s,a) = old value + learning rate * (reward + discount rate * bestCaseQ of futrure)

"""

episodes = 5000
epsilon = 0.5
discountRate = 0.95
learningRate = 0.1
renderStep = 5001
epsilonMax = 1
epsilonStep = (epsilonMax - epsilon)/episodes

steps = 8

possibleStates = [steps] * len(env.observation_space.high) # i think that 20 is way to low for now but we'll see
stateLengths = (env.observation_space.high-env.observation_space.low) / steps

qLibrary = np.random.uniform(low = -5, high = 0, size = (possibleStates) + [env.action_space.n])

def getCleanedState(yuckyState):
    indices = (((yuckyState-env.observation_space.low)/stateLengths).astype(int))
    return tuple(np.clip(indices,0,steps-1))

attemptTracking = {"ep": [], "successRate": []} #episode --> succes  rate
successes = 0

for episode in range(episodes):

    env.close()
    # env = gym.make(
    #     "Acrobot-v1",
    #     render_mode="human" if episode % renderStep == 0 else None,
    # )

    if (episode % 1000 == 0):
        print(episode)
    env = gym.make(
        "Acrobot-v1",
        render_mode=None,
    )

    observationState,information = env.reset()
    observationState = getCleanedState(observationState)


    done = False

    while (not done):
        if np.random.random() > epsilon: 
            action = np.argmax(qLibrary[observationState]) # 0,1,2 
        else:
            action = random.randint(0,env.action_space.n-1)

        qstep = (qLibrary[observationState])[action] #some value

        newState, reward, terminated, truncated, info =  env.step(action) #is action in the correct scope 
        newState = getCleanedState(newState)

        done = terminated or truncated

        if not terminated:
            upcomingQMax = np.max(qLibrary[newState])
            qLibrary[observationState][action] = (1-learningRate) * qstep +  learningRate  * (reward + discountRate * upcomingQMax)
        elif terminated:
            qLibrary[observationState][action] = 0
            successes += 1

        attemptTracking["ep"].append(episode)
        attemptTracking["successRate"].append(100 * successes / (episode + 1))
        observationState = newState

    epsilon -= (epsilonStep)

env.close()
mpl.plot(attemptTracking["ep"],attemptTracking["successRate"], label = "success rate")
mpl.legend(loc = 4)
mpl.show()

