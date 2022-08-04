import numpy as np
import scipy as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import copy
import torch


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
    
class ETA(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ETA, self).__init__()
        self.fc_1 = nn.Linear(state_dim + action_dim, 64)
        self.fc_2 = nn.Linear(64, 64)
        self.fc_out = nn.Linear(64, 1)
        
    def forward(self, s, a):
        eta = F.relu(self.fc_1(torch.cat([s, a], 1)))
        eta = F.relu(self.fc_2(eta))
        eta = self.fc_out(eta)
        return eta.squeeze(dim=1)
        
    def save(self, filename):
        torch.save(self.state_dict(), filename)
    
    def load(self, filename):
        torch.load_state_dict(torch.load(filename))      

class RFQI(object):
    def __init__(self, state_dim, action_dim, min_action, max_action, device, 
                 env_action_type, adam_lr=3e-4, adam_eps=1e-6, actor_lr=6e-4, 
                 critic_lr=1e-3, gamma=0.99, tau=0.005, lmbda=0.75, phi=0.05,
                 rho=0.5):
        latent_dim = action_dim * 2
        # important - need a tensor to clamp actions at:
        # 1. Actor action forward
        # 2. VAE action (visitation) output
        # 3. model predict action
        self.max_action = torch.tensor(max_action, dtype=torch.float, 
                                       device=device)
        self.min_action = torch.tensor(min_action, dtype=torch.float, 
                                       device=device)
        # learning rates*
        self.adam_lr = adam_lr
        self.adam_eps = adam_eps
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        
        # initialize
        self.actor = Actor(state_dim, action_dim, self.max_action, phi).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                eps=self.adam_eps,
                                                lr=actor_lr)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 eps=self.adam_eps,
                                                 lr=critic_lr)

        self.vae = VAE(state_dim, action_dim, action_dim * 2, self.max_action, device).to(device)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters())
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.lmbda = lmbda
        self.rho = rho
        self.device = device
        self.env_action_type = env_action_type
        
        # eta limit
        self.eta_low, self.eta_high = 0, 1 / (self.rho * (1 - self.gamma))
        

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).repeat(100, 1).to(self.device)
            action = self.actor(state, self.vae.decode(state))
            q1 = self.critic.q1(state, action)
            ind = q1.argmax(0)
            res = action[ind]
            res = torch.clamp(res, self.min_action, self.max_action)
        return res.cpu().data.numpy().flatten()
    
    def predict(self, state):
        '''
        RFQI version of predict.
        '''
        action = self.select_action(state)
        if self.env_action_type == 'discrete':
            action = np.rint(action[0]).astype(int)
        elif self.env_action_type == 'continuous':
            action = action[0]
        else:
            raise NotImplementedError
        return action
    

    def optimize_eta(self, V_ns, s, a, writer=None, log_step=None, tol=1e-3, 
                     max_iter=int(1e4)):
        eta_func = ETA(self.state_dim, self.action_dim).to(self.device)
        eta_optimizer = torch.optim.Adam(eta_func.parameters(), 
                                         lr=self.adam_lr,
                                         eps=self.adam_eps,
                                         maximize=True)
        
        def g(s, a, V_ns, rho, eta_func):
            val = 0
            eta = eta_func(s, a)
            g_val = -torch.maximum(eta - V_ns, s.new_tensor(0))
            g_val += (1 - rho) * eta
            g_val = g_val.sum()
            return g_val

        # optimize
        prev_gval = torch.tensor([float('inf')]).to(self.device)
        curr_loss = None
        for i in range(max_iter):
            gval = g(s, a, V_ns, self.rho, eta_func)
            eta_optimizer.zero_grad()
            gval.backward()
            eta_optimizer.step()
            loss = torch.norm(gval - prev_gval)
            curr_loss = loss
            if loss < tol:
                break
            else:
                prev_gval = gval
        if writer is not None:
            writer.add_scalar("eta_loss", curr_loss.detach().cpu().numpy(),
                              log_step)
            writer.add_scalar('eta_opt_num_of_steps', i, log_step)
            writer.flush()
        etas = eta_func(s,a).detach()
        etas = torch.clamp(etas, self.eta_low, self.eta_high).unsqueeze(dim=1)

        return eta_func, etas
        

    def train(self, data, trn_steps, batch_size=100, writer=None,
              log_base=0):
#         ps = torch.full((N,), 1/N, dtype=torch.float, device=self.device)
        gamma = torch.tensor(self.gamma, dtype=torch.float, device=self.device)
        rho = torch.tensor(self.rho, dtype=torch.float, device=self.device)

        for i in range(trn_steps):
            # Sample replay buffer / batch
            state, action, next_state, reward, not_done = data.sample(batch_size)

            # Variational Auto-Encoder Training
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
                next_state = torch.repeat_interleave(next_state, 10, 0)

                # Compute value of perturbed actions sampled from the VAE
                target_Q1, target_Q2 = self.critic_target(next_state, self.actor_target(next_state, self.vae.decode(next_state)))

                # Soft Clipped Double Q-learning
                target_Q = self.lmbda * torch.min(target_Q1, target_Q2) + (1. - self.lmbda) * torch.max(target_Q1, target_Q2)
                
                # Take max over each action sampled from the VAE
                target_Q = target_Q.reshape(batch_size, -1).max(1)[0].reshape(-1, 1)
            
            # g optimize
            # clock it
            start = time.time()
            _, etas = self.optimize_eta(target_Q, state, action,
                                        log_step=i+log_base,
                                        writer=writer)
            end = time.time()
            test = etas - target_Q
            # critic training cont.
            with torch.no_grad():     
                # robust-fqi target
                target_Q = reward - gamma * torch.maximum(etas - target_Q, etas.new_tensor(0)) + (1 - rho) * etas * gamma
                
            current_Q1, current_Q2 = self.critic(state, action)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Pertubation Model / Action Training
            sampled_actions = self.vae.decode(state)
            perturbed_actions = self.actor(state, sampled_actions)

            # Update through DPG
            actor_loss = -self.critic.q1(state, perturbed_actions).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update Target Networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            # log tensorboard
            if writer is not None:
                writer.add_scalar("critic_loss", critic_loss, i+log_base)
                writer.add_scalar("actor_loss", actor_loss, i+log_base)
                writer.add_scalar("max eta", max(etas), i+log_base)
                writer.add_scalar("clocktime to optimize g(eta)", end-start, 
                                  i+log_base)
        writer.flush()
