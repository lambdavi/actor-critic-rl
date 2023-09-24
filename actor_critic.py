import torch
from networks import ActorCriticNetwork

class Agent:
    def __init__(self, alpha=0.0003, gamma=0.99, n_actions=2) -> None:
        self.gamma = gamma
        self.n_actions = n_actions
        self.action = None
        self.action_space = [i for i in range(self.n_actions)]

        self.model = ActorCriticNetwork(n_actions=n_actions)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=alpha)

        self.model.train()
    
    def choose_action(self, observation):
        state = torch.tensor([observation])
        _, probs = self.model(state)
        action_probabilities = torch.distributions.Categorical(probs=probs)
        action = action_probabilities.sample()
        self.action = action
        print(self.action)
        return action.numpy()[0]

    def save_models(self) -> None:
        print('... saving models ...')
        torch.save(self.model.state_dict(), self.model.checkpoint_file)

    def load_models(self) -> None:
        print('... loading model ...')
        state_dict = torch.load(self.model.checkpoint_file)
        self.model.load_state_dict(state_dict)
    
    def learn(self, state, reward, state_, done):
        """
            State: current state
            Reward: future reward
            State_: next state
            Done: is it done learning?
        """
        state = torch.tensor([state], dtype=torch.float32)
        state_ = torch.tensor([state_], dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)
        
        self.optimizer.zero_grad()

        state_value, probs = self.model(state)
        state_value_, _ = self.model(state_)
        state_value = torch.squeeze(state_value)
        state_value_ = torch.squeeze(state_value_)

        action_probs = torch.distributions.Categorical(probs = probs)
        log_prob = action_probs.log_prob(self.action)

        # Temporal difference delta (0 future value if done)
        delta = reward + self.gamma*state_value_(1-int(done)) - state_value
        actor_loss = -log_prob*delta
        critic_loss = delta ** 2

        total_loss = torch.tensor(actor_loss + critic_loss)
        total_loss.backward()

        self.optimizer.step()

        


