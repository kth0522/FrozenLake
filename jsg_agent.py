import numpy as np
import gym
import time

class Agent:
    def __init__(self, Q, env, mode):
        self.Q = Q
        self.env = env
        self.mode = mode

        # Hyperparameter
        self.alpha = 0.01
        self.max_alpha = 0.01
        self.min_alpha = 0.001
        self.decay_rate = 0.001

        self.gamma = 0.4

        self.max_episode = 20000
        self.max_iteration = 10000

        self.epsilon = 0.001
        self.max_epsilon = 0.001
        self.min_epsilon = 0.0001


        self.state_size = env.observation_space.n
        self.action_size = env.action_space.n

        if mode == "learning_mode":
            # Initializing Q
            for i in range(self.state_size):
                self.Q[i]

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
            action = np.argmax(self.Q[state])
        else:
            action = np.random.choice(self.actions)

        return action

    def update_Q(self, state, new_state, reward, action):
        self.Q[state][action] = (1-self.alpha)*self.Q[state][action]+self.alpha*(reward+self.gamma*np.max(self.Q[new_state]))

    def learn(self):
        rewards = []
        for cur_episode in range(self.max_episode):
            state = self.env.reset()
            episode_reward = 0
            done = False
            step = 0

            # While episode is terminated
            while not done:
                self.epsilon = self.min_epsilon + (self.max_epsilon-self.min_epsilon)*np.exp(-self.decay_rate * step)
                self.alpha = self.min_alpha + (self.max_alpha-self.min_epsilon)*np.exp(-self.decay_rate * step)

                action = self.select_action(state)
                new_state, reward, done, _ = self.env.step(action)

                if reward == 0:
                    if done:
                        if state == self.state_size-1:
                            reward = 1
                        else:
                            reward = -1
                    else:
                        reward = -0.01


                self.update_Q(state, new_state, reward, action)

                state = new_state
                episode_reward += reward
                step += 1


            rewards.append(episode_reward)



            if cur_episode%100 == 0:
                avg_reward = sum(rewards) / self.max_episode
                print("\rEpisode {}/{} || average reward {}".format(cur_episode, self.max_episode, avg_reward), end="")




