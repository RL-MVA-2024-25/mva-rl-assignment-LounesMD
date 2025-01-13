from gymnasium.wrappers import TimeLimit
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import os 

from fast_env import FastHIVPatient
from interface import Agent
from tqdm import tqdm

class ProjectAgent(Agent):
    def __init__(self, state_dim=6, action_dim=4, gamma=0.99, max_iterations=400, epsilon_mean=0.005, epsilon_infinity=0.05):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.max_iterations = max_iterations
        self.epsilon_mean = epsilon_mean
        self.epsilon_infinity = epsilon_infinity
        self.q_model = xgb.XGBRegressor(
            booster='gbtree',
            learning_rate=0.1,
            max_depth=6,
            n_estimators=200,
        )

    def fit(self, transitions, eval_env=None, eval_episodes=5):
        """
        Transitions: List of tuples [(state, action, reward, next_state), ...]
        """
        states, actions, rewards, next_states = zip(*transitions)
        states = np.array(states)
        actions = np.array(actions).reshape(-1, 1)
        rewards = np.array(rewards)
        next_states = np.array(next_states)

        for iteration in range(self.max_iterations + 1):
            inputs = np.hstack((states, actions))
            if iteration > 0:
                next_q_values = self.q_model.predict(np.hstack([
                    np.repeat(next_states, self.action_dim, axis=0),
                    np.tile(np.arange(self.action_dim).reshape(-1, 1), (len(next_states), 1))
                ])).reshape(len(next_states), self.action_dim)
                max_next_q_values = np.max(next_q_values, axis=1)
            else:
                max_next_q_values = np.zeros(len(next_states))

            targets = rewards + self.gamma * max_next_q_values
            X_train, X_val, y_train, y_val = train_test_split(inputs, targets, test_size=0.2)
            self.q_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],                
                verbose=False
            )

            if eval_env and ((iteration) % 50 == 0):
                eval_rewards = self.evaluate_policy(eval_env, eval_episodes)
                print(f"Evaluation Reward after Iteration {iteration + 1}: {np.mean(eval_rewards)}")

    def act(self, state):
        """
        Select the best action for a given state.
        """
        state_repeated = np.repeat(state.reshape(1, -1), self.action_dim, axis=0)
        actions = np.arange(self.action_dim).reshape(-1, 1)
        inputs = np.hstack((state_repeated, actions))
        q_values = self.q_model.predict(inputs)
        return np.argmax(q_values)

    def evaluate_policy(self, env, episodes=5):
        """
        Evaluate the policy by running it on the environment.
        Returns the total rewards for the episodes.
        """
        total_rewards = []
        for _ in range(episodes):
            obs = env.reset()[0]
            episode_reward = 0
            for _ in range(200):
                action = self.act(obs)
                obs, reward, _, _, _ = env.step(action)
                episode_reward += reward
            total_rewards.append(episode_reward)
        return total_rewards

    def save(self, filepath="q_model.json"):
        """
        Save the XGBoost model to a file.
        """
        self.q_model.save_model(filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath="q_model.json"):
        """
        Load the XGBoost model from a file.
        """
        if os.path.exists(filepath):
            self.q_model.load_model(filepath)
            print(f"Model loaded from {filepath}")
        else:
            raise FileNotFoundError(f"No model found at {filepath}")

# Train the agent
if __name__ == "__main__":
    fast_env = TimeLimit(env=FastHIVPatient(domain_randomization=True), max_episode_steps=200)

    agent = ProjectAgent(state_dim=6, action_dim=4)
    train = False
    if train:
        transitions = []
        for _ in range(30):
            obs = fast_env.reset()[0]
            for _ in range(200):
                action = np.random.randint(0, 4)
                next_obs, reward, _,_,_ = fast_env.step(action)
                transitions.append((obs, action, reward, next_obs))        
                obs = next_obs

        for _ in tqdm(range(11)):
            res = agent.fit(transitions, eval_env=fast_env, eval_episodes=5)
            for _ in range(30):
                obs = fast_env.reset()[0]
                for _ in range(200):
                    action = agent.act(obs)
                    next_obs, reward, _,_,_ = fast_env.step(action)
                    transitions.append((obs, action, reward, next_obs))        
                    obs = next_obs
    
    agent.save("q_model.json")
