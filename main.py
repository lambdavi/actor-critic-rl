import gym
import numpy as np
import time
from actor_critic import Agent
from utils import plot_learning_curve

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    agent = Agent(alpha=5e-4, n_actions=env.action_space.n)
    n_games = 1000

    filename = 'cartpole.png'
    figure_file = 'plots/' + filename

    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()
    
    for i in range(n_games):
        observation = env.reset()[0]
        done = False
        score = 0

        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info, _ = env.step(action)
            #env.render()

            score += reward
            if not load_checkpoint:
                agent.learn(observation, reward, observation_, done)
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()
        if i%100==0:
            print(f"Episode: {i}, score: {best_score}, avg_score {avg_score}")

    
    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, score_history, figure_file)
