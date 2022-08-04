import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def add_gaussian_noise(actions, max_action, std):
    return (
        actions
        + max_action * std * torch.randn_like(actions)
    ).clamp(-max_action, max_action)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, phi=0.05):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action
        self.phi = phi

    def forward(self, state, action):
        a = F.relu(self.l1(torch.cat([state, action], 1)))
        a = F.relu(self.l2(a))
        a = self.phi * self.max_action * torch.tanh(self.l3(a))
        return (a + action).clamp(-self.max_action, self.max_action)
    
    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename, device='cpu'):
        self.to(device)
        self.load_state_dict(torch.load(filename, 
                                        map_location=torch.device(device)))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        self.l4 = nn.Linear(state_dim + action_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)

    def forward(self, state, action):
        q1 = F.relu(self.l1(torch.cat([state, action], 1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(torch.cat([state, action], 1)))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def q1(self, state, action):
        q1 = F.relu(self.l1(torch.cat([state, action], 1)))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1
    
    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename, device='cpu'):
        self.to(device)
        self.load_state_dict(torch.load(filename, 
                                        map_location=torch.device(device)))
    
# Vanilla Variational Auto-Encoder 
class VAE(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, max_action, device):
        super(VAE, self).__init__()
        self.e1 = nn.Linear(state_dim + action_dim, 750)
        self.e2 = nn.Linear(750, 750)

        self.mean = nn.Linear(750, latent_dim)
        self.log_std = nn.Linear(750, latent_dim)

        self.d1 = nn.Linear(state_dim + latent_dim, 750)
        self.d2 = nn.Linear(750, 750)
        self.d3 = nn.Linear(750, action_dim)

        self.max_action = max_action
        self.latent_dim = latent_dim
        self.device = device

    def forward(self, state, action):
        z = F.relu(self.e1(torch.cat([state, action], 1)))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)

        u = self.decode(state, z)

        return u, mean, std

    def decode(self, state, z=None):
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        if z is None:
            z = torch.randn((state.shape[0], self.latent_dim)).to(self.device).clamp(-0.5, 0.5)

        a = F.relu(self.d1(torch.cat([state, z], 1)))
        a = F.relu(self.d2(a))
        return self.max_action * torch.tanh(self.d3(a))
    
    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename, device='cpu'):
        self.to(device)
        self.load_state_dict(torch.load(filename, 
                                        map_location=torch.device(device)))


class VAE_state(nn.Module):
    def __init__(self, state_dim, latent_dim, max_state, device):
        super(VAE_state, self).__init__()
        self.e1 = nn.Linear(state_dim, 750)
        self.e2 = nn.Linear(750, 750)

        self.mean = nn.Linear(750, latent_dim)
        self.log_std = nn.Linear(750, latent_dim)

        self.d1 = nn.Linear(latent_dim, 750)
        self.d2 = nn.Linear(750, 750)
        self.d3 = nn.Linear(750, state_dim)

        self.max_state = max_state
        self.latent_dim = latent_dim
        self.device = device

    def forward(self, state):
        z = F.relu(self.e1(state))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)

        u = self.decode(state.shape[0], z)

        return u, mean, std

    def decode(self, batch_size, z=None):
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        if z is None:
            z = torch.randn((batch_size, self.latent_dim)).to(self.device).clamp(-0.5, 0.5)

        s = F.relu(self.d1(z))
        s = F.relu(self.d2(s))
        if self.max_state is None:
            return self.d3(s)
        else:
            return self.max_state * torch.tanh(self.d3(s))

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename, device='cpu'):
        self.to(device)
        self.load_state_dict(torch.load(filename, 
                                        map_location=torch.device(device))) 

class PQL_BCQ(object):
    def __init__(self, state_dim, action_dim, max_state, min_action, max_action,
                 device, discount=0.99, tau=0.005, lmbda=0.75,
                 phi=0.05, n_action=100, n_action_execute=100, 
                 backup="QL", ql_noise=0.0,
                 actor_lr=1e-3, beta=-0.4, vmin=0):
        # Add on - need a tensor to clamp actions at:
        # 1. Gaussian noise
        # 2. Actor action forward
        # 3. VAE action (visitation) output
        # 4. model predict action
        self.max_action = torch.tensor(max_action, dtype=torch.float, 
                                       device=device)
        self.min_action = torch.tensor(min_action, dtype=torch.float, 
                                       device=device)
        
        self.actor = Actor(state_dim, action_dim, self.max_action, phi).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

        # define VAE action (visitation)
        # latent_dim = action_dim * 2
        self.vae = VAE(state_dim, action_dim, action_dim * 2, self.max_action, device).to(device)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters())

        # define VAE state (visitation)
        # latent_dim = state_dim * 2
        self.vae2 = VAE_state(state_dim, state_dim * 2, max_state, device).to(device)
        self.vae2_optimizer = torch.optim.Adam(self.vae2.parameters())

        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.lmbda = lmbda
        self.device = device
        self.beta = beta
        self.n_action = n_action
        self.n_action_execute = n_action_execute
        self.backup = backup
        self.ql_noise = ql_noise
        self.vmin = vmin

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).repeat(self.n_action_execute, 1).to(self.device)
            action = self.actor(state, self.vae.decode(state))
            q1 = self.critic.q1(state, action)
            ind = q1.argmax(0)
            # add on properly clamp output action
            res = action[ind]
            res = torch.clamp(res, self.min_action, self.max_action)
        return res.cpu().data.numpy().flatten()

    def train_vae(self, replay_buffer, iterations, batch_size=100):
        scores = []
        for it in range(iterations):
            # Sample replay buffer / batch
            state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

            recon, mean, std = self.vae2(state)
            recon_loss = F.mse_loss(recon, state)
            KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
            vae_loss = recon_loss + 0.5 * KL_loss
            scores.append(vae_loss.item())

            self.vae2_optimizer.zero_grad()
            vae_loss.backward()
            self.vae2_optimizer.step()

            recon, mean, std = self.vae2(next_state)
            recon_loss = F.mse_loss(recon, next_state)
            KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
            vae_loss = recon_loss + 0.5 * KL_loss
            scores.append(vae_loss.item())

            self.vae2_optimizer.zero_grad()
            vae_loss.backward()
            self.vae2_optimizer.step()
        return np.mean(scores)

    def train_action_vae(self, replay_buffer, iterations, batch_size=100):
        for it in range(iterations):
            state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

            # Variational Auto-Encoder Training
            recon, mean, std = self.vae(state, action)
            recon_loss = F.mse_loss(recon, action)
            KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
            vae_loss = recon_loss + 0.5 * KL_loss

            self.vae_optimizer.zero_grad()
            vae_loss.backward()
            self.vae_optimizer.step()

    def test_vae(self, replay_buffer, batch_size=1000):
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        recon, mean, std = self.vae2(next_state)
        recon_loss = ((recon - next_state) ** 2).mean(dim=1)
        KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean(dim=1)
        vae_loss = recon_loss + 0.5 * KL_loss
        return -vae_loss.detach().cpu().numpy()

    def train(self, replay_buffer, iterations, batch_size=100):

        mean_scores = []

        for it in range(iterations):
            # Sample replay buffer / batch
            state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

            # Training action vae
            recon, mean, std = self.vae(state, action)
            recon_loss = F.mse_loss(recon, action)
            KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
            vae_loss = recon_loss + 0.5 * KL_loss

            self.vae_optimizer.zero_grad()
            vae_loss.backward()
            self.vae_optimizer.step()

            # Critic Training
            with torch.no_grad():
                # Duplicate next state 10 times
                next_state = torch.repeat_interleave(next_state, self.n_action, 0)

                # Compute value of perturbed actions sampled from the VAE
                if self.backup == "QL":
                    target_Q1, target_Q2 = self.critic_target(next_state, add_gaussian_noise(self.vae.decode(next_state), self.max_action, self.ql_noise))
                else:
                    target_Q1, target_Q2 = self.critic_target(next_state, self.actor_target(next_state, self.vae.decode(next_state)))
                # Soft Clipped Double Q-learning
                target_Q = self.lmbda * torch.min(target_Q1, target_Q2) + (1. - self.lmbda) * torch.max(target_Q1,
                                                                                                        target_Q2)
                # Take max over each action sampled from the VAE
                target_Q = target_Q.reshape(batch_size, -1).max(1)[0].reshape(-1, 1)
                if self.beta < 0:
                    recon, mean, std = self.vae2(next_state)
                    recon_loss = ((recon - next_state) ** 2).mean(dim=1)
                    KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean(dim=1)
                    score = -recon_loss - 0.5 * KL_loss
                    score = score.reshape(batch_size, -1).mean(dim=1, keepdim=True)
                    score = torch.sigmoid(100 * (score - self.beta))
                    mean_scores.append(score.mean().item())
                else:
                    score = 1
                target_Q = reward + not_done * score * self.discount * target_Q + not_done * self.discount * (1 - score) * self.vmin

            current_Q1, current_Q2 = self.critic(state, action)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Pertubation Model / Action Training
            sampled_actions = self.vae.decode(state)
            perturbed_actions = self.actor(state, sampled_actions)

            # Update through DPG
            actor_loss = -(self.critic.q1(state, perturbed_actions)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update Target Networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # print(score)
        print("Average state filter score:", np.mean(mean_scores))

        return mean_scores
