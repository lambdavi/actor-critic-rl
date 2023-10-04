import gym
import numpy as np
from agents.actor_critic import Agent
from utils import plot_learning_curve

def run_ac():
    eval_mode = False
    if eval_mode:
        env = gym.make('CartPole-v1', render_mode="human")
    else:
        env = gym.make('Acrobot-v1', render_mode="human")
    load_checkpoint = False
    agent = Agent(alpha=5e-4, obs_space=env.observation_space.shape[0], n_actions=env.action_space.n, eval_mode=eval_mode)
    n_games = 1000

    filename = f'{env.spec.name}.png' 
    figure_file = 'plots/' + filename

    best_score = env.reward_range[0]
    best_score_greedy = env.reward_range[0]
    score_history = []
    
    if load_checkpoint:
        agent.load_models("_best")
    
    for i in range(n_games):
        observation = env.reset()[0]
        done = False
        score = 0
        
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info, _ = env.step(action)
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
        if score > best_score_greedy:
            best_score_greedy = score
            if not load_checkpoint:
                agent.save_models("_greedy")

        if i%10==0:
            print(f"Episode: {i}, score: {score}, avg_score {avg_score}, best_score: {best_score}, best_score_g: {best_score_greedy}")

    
    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, score_history, figure_file)
