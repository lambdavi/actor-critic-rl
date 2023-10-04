import numpy as np
import torch
from torch.optim import Adam
from buffer import ReplayBuffer
from models.ddpg_network import ActorNetwork, CriticNetwork

class DDPG_Agent():
    def __init__(self, input_dims, obs_space, alpha=0.001, beta=0.002, env=None,
                 gamma = 0.95, n_actions=2, max_size=1000000, tau=.005,
                 fc1=400, fc2=300, batch_size=64, noise=0.1) -> None:
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.noise = noise
        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]

        self.actor = ActorNetwork(obs_space=obs_space, n_actions=self.n_actions, name='actor')
        self.critic = CriticNetwork(obs_space=obs_space, name='critic')
        self.target_actor = ActorNetwork(obs_space=obs_space, n_actions=self.n_actions, name='target_actor')        
        self.critic = CriticNetwork(obs_space=obs_space, name='critic')
        self.target_critic = CriticNetwork(obs_space=obs_space, name='target_critic')

        self.actor_optimizer = Adam(lr=alpha)
        self.critic_optimizer = Adam(lr=beta)
        self.target_actor_optimizer = Adam(lr=alpha)
        self.target_critic_optimizer = Adam(lr=beta)

        self.loss = torch.nn.MSELoss()
        self.update_network_parameters(tau=1)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        
        # Update target actor
        weights = []
        for target, weight in zip(self.target_actor.parameters(), self.actor.parameters()):
            if param.requires_grad:  # Check if the parameter is trainable
                weights.append(weight*tau + target*(1-tau))

        for i, param in enumerate(self.target_actor.parameters()):
            if param.requires_grad:  # Check if the parameter is trainable
                param.data.fill_(weights[i])  # Set all parameter values to 0.0

        # Update target critc
        weights = []
        for target, weight in zip(self.target_critic.parameters(), self.critic.parameters()):
            if param.requires_grad:  # Check if the parameter is trainable
                weights.append(weight*tau + target*(1-tau))

        for i, param in enumerate(self.target_critic.parameters()):
            if param.requires_grad:  # Check if the parameter is trainable
                param.data.fill_(weights[i])  # Set all parameter values to 0.0

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)


    def save_models(self, name="") -> None:
        #print('... saving models ...')
        torch.save(self.actor.state_dict(), self.actor.checkpoint_file+name)
        torch.save(self.critic.state_dict(), self.critic.checkpoint_file+name)
        torch.save(self.target_actor.state_dict(), self.target_actor.checkpoint_file+name)
        torch.save(self.target_critic.state_dict(), self.target_critic.checkpoint_file+name)

    def load_models(self, name="") -> None:
        print('... loading models ...')
        a_state_dict = torch.load(self.actor.checkpoint_file+name)
        self.actor.load_state_dict(a_state_dict)

        c_state_dict = torch.load(self.critic.checkpoint_file+name)
        self.critic.load_state_dict(c_state_dict)

        ta_state_dict = torch.load(self.target_actor.checkpoint_file+name)
        self.target_actor.load_state_dict(ta_state_dict)

        tc_state_dict = torch.load(self.target_critic.checkpoint_file+name)
        self.target_critic.load_state_dict(tc_state_dict)

    def choose_action(self, observation, evaluate=False):
        state = torch.tensor([observation], dtype=torch.float32)
        with torch.no_grad():
            actions = self.actor(state)

        if not evaluate:
            actions += 0.0 + self.noise*torch.rand(self.n_actions, 1)
        
        actions = torch.clamp(actions, self.min_action, self.max_action)

        return actions[0]
    
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.actor.train()
        self.critic.train()
        self.target_actor.train()
        self.target_critic.train()

        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        states = torch.tensor(np.array([state]), dtype=torch.float32)
        states_ = torch.tensor(np.array([new_state]), dtype=torch.float32)
        actions = torch.tensor(np.array([action]), dtype=torch.float32)
        rewards = torch.tensor(reward, dtype=torch.float32)

        # Start training
        self.critic_optimizer.zero_grad()
        self.target_actor_optimizer.zero_grad()
        self.target_critic_optimizer.zero_grad()

        target_actions = self.target_actor(states_)
        critic_value_ = torch.squeeze(self.target_critic(states_, target_actions), dim=1)
        critic_value = torch.squeeze(self.critic(states, actions), 1)
        target = rewards + self.gamma*critic_value_*(1-done)
        critic_loss = self.loss(target, critic_value)
        
        critic_loss.backward()

        self.critic_optimizer.step()
        self.target_actor.step()
        self.target_critic.step()

        self.actor_optimizer.zero_grad()

        new_policy_actions = self.actor(states)
        actor_loss = -self.critic(states, new_policy_actions) # Gradient ascent
        actor_loss = torch.mean(actor_loss, requires_grad=True)

        actor_loss.backward()
        self.actor_optimizer.step()

        self.update_network_parameters()








