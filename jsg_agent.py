import numpy as np
import gym
import time

class Agent:
    def __init__(self, Q, env, mode):
        self.Q1 = Q
        self.Q2 = Q
        self.env = env
        self.mode = mode

        # Hyperparameter
        self.alpha = 0.01
        self.gamma = 0.9

        self.max_episode = 1000
        self.max_iteration = 10000

        self.epsilon = 0.001

        self.state_size = env.observation_space.n
        self.action_size = env.action_space.n

        if mode == "learning_mode":
            # Initializing Q table
            for i in range(self.state_size):
                self.Q1[i]
                self.Q2[i]
        elif mode == "testing_mode":
            # If it is in testing mode, just use the existing Q table
            pass
        else:
            raise NameError('Use valid mode')

        # Possible actions list
        self.actions = np.arange(self.action_size)

    # Epsilon-greedy exploration
    def select_action(self, state):
        if np.random.rand() > self.epsilon:
            action = np.argmax(self.Q1[state]+self.Q2[state])
        else:
            action = np.random.choice(self.actions)

        return action

    # Double Q-Learning
    def update_Q(self, state, new_state, reward, action):
        if np.random.rand() > 0.5:
            self.Q1[state][action] = self.Q1[state][action] + self.alpha*(reward+self.gamma*self.Q2[new_state][np.argmax(self.Q1[new_state])]-self.Q1[state][action])
        else:
            self.Q2[state][action] = self.Q2[state][action] + self.alpha * (
                        reward + self.gamma * self.Q1[new_state][np.argmax(self.Q2[new_state])] - self.Q2[state][action])
    def learn(self):
        rewards = []
        for cur_episode in range(self.max_episode):
            state = self.env.reset()
            episode_reward = 0
            step = 0
            done = False

            # While episode is terminated
            while not done:
                action = self.select_action(state)
                new_state, reward, done, _ = self.env.step(action)
1
                if done:
                    if state == self.state_size-1:
                        reward = +1
                    else:
                        reward = -1
                else:
                    reward = -0.01

                self.update_Q(state, new_state, reward, action)

                state = new_state
                episode_reward += reward
                step+=1

                if done:
                    break


            rewards.append(episode_reward)



            if cur_episode%100 == 0:
                avg_reward = sum(rewards) / self.max_episode
                print("\rEpisode {}/{} || average reward {}".format(cur_episode, self.max_episode, avg_reward), end="")




