import numpy as np
import gym

class Agent:
    def __init__(self, Q, env, mode):
        self.Q = Q
        self.env = env
        self.mode = mode

        #Hyperparameter
        self.alpha = 0.1
        self.gamma = 0.8
        self.epsilon = 0.1
        self.max_episode = 1000

        self.state_size = env.observation_space.n
        self.action_size = env.action_space.n

        if mode == "learning_mode":
            #Initializing Q table
            for i in range(self.state_size):
                self.Q[i]
        elif mode == "testing_mode":
            #If it is in testing mode, just use the existing Q table
            pass
        else:
            raise NameError('Use valid mode')

        self.actions = np.arange(self.action_size)

    #Epsilon-greedy exploration
    def select_action(self, state):
        if np.random.rand() > self.epsilon:
            action = np.argmax(self.Q[state])
        else:
            action = np.random.choice(self.actions)

        return action

    def learn(self):
        rewards = []
        for cur_episode in range(self.max_episode):
            state = self.env.reset()
            episode_reward = 0


            #While episode is terminated
            while True:
                action = self.select_action(state)
                new_state, reward, done, _ = self.env.step(action)

                self.Q[new_state][action] = self.Q[new_state][action] + self.alpha*(reward + self.gamma*np.max(self.Q[new_state])-self.Q[state][action])
                episode_reward += reward
                if done:
                    if reward == 0:
                        reward = -1
                    rewards.append(episode_reward)
                    self.Q[new_state][action] = self.Q[new_state][action] + self.alpha * (
                                reward + self.gamma * np.max(self.Q[new_state]) - self.Q[state][action])
                    break
                else:
                    if reward == 0:
                        reward = -0.1
                    rewards.append(episode_reward)
                    self.Q[new_state][action] = self.Q[new_state][action] + self.alpha * (
                            reward + self.gamma * np.max(self.Q[new_state]) - self.Q[state][action])
                state = new_state


            if cur_episode%100 == 0:
                avg_reward = sum(rewards) / self.max_episode
                print("\rEpisode {}/{} || average reward {}".format(cur_episode, self.max_episode, avg_reward), end="")




