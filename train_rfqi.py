import numpy as np
import torch
import time
import gym
import os
import imageio
from gym import spaces
import argparse
from torch.utils.tensorboard import SummaryWriter

from data_container import DATA
from rfqi import RFQI

def get_action_type(action_space):
    """
    Method to get the action type to choose prob. dist. 
    to sample actions from NN logits output.
    """
    if isinstance(action_space, spaces.Box):
        shape = action_space.shape
        assert len(shape) == 1
        if shape[0] == 1:
            return 'continuous'
        else:
            return 'multi_continuous'
    elif isinstance(action_space, spaces.Discrete):
        return 'discrete'
    elif isinstance(action_space, spaces.MultiDiscrete):
        return 'multi_discrete'
    elif isinstance(action_space, spaces.MultiBinary):
        return 'multi_binary'
    else:
        raise NotImplementedError


def eval_policy(policy, env_name, eval_episodes=10):
    rewards = []
    env = gym.make(env_name)
    env_action_type = get_action_type(env.action_space)
    for _ in range(eval_episodes):
        state, done = env.reset(seed=np.random.randint(100000)), False
        eps_reward = 0.0
        while not done:
            action = policy.select_action(np.array(state))
            if env_action_type == 'discrete':
                action = np.rint(action[0]).astype(int)
            elif env_action_type == 'continuous':
                action = action[0]
            else:
                pass
            if 'FrozenLake' in env_name:
                action = int(action)
            state, reward, done, _ = env.step(action)
            eps_reward += reward
        rewards.append(eps_reward)
    avg, std = np.average(rewards), np.std(rewards)
    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes")
    print(rewards)
    print("---------------------------------------")
    return avg, std

def generate_gif(policy, env_name, video_path):
    images = []
    env = gym.make(env_name)
    state = env.reset()
    img = env.render(mode='rgb_array')
    done = False
    while not done:
        images.append(img)
        action = policy.select_action(np.array(state))
        if env_action_type == 'discrete':
            action = np.rint(action[0]).astype(int)
        elif env_action_type == 'continuous':
            action = action[0]
        else:
            pass
        state, reward, done, info = env.step(action)
        img = env.render(mode='rgb_array')
    imageio.mimsave(video_path, 
                    [np.array(img) for i, img in enumerate(images) if i%2 == 0], 
                    fps=29)
    
def save_policy(policy, save_path):
    policy.actor.save(f'{save_path}_actor')
    policy.critic.save(f'{save_path}_critic')
    policy.vae.save(f'{save_path}_vae')

def train_rfqi(state_dim, action_dim, min_action, max_action, paths,
               env_action_type, args):
    # prep overhead
    # parse paths
    data_path = paths['data_path']
    log_path = f"./logs/{paths['save_path']}"
    save_path = f"./models/{paths['save_path']}"
    video_path = f"./videos/{paths['save_path']}.gif"
    
    # dataset
    data = DATA(state_dim, action_dim, args.device)
    data.load(data_path, args.data_size)
    writer = SummaryWriter(log_path)
    # initialize policy
    policy = RFQI(state_dim, action_dim, min_action, max_action, args.device,
                  env_action_type, adam_lr=args.adam_lr, adam_eps=args.adam_eps,
                  actor_lr=args.actor_lr, critic_lr=args.critic_lr,
                  rho=args.rho, gamma=args.gamma, tau=args.tau, 
                  lmbda=args.lmbda, phi=args.phi)
        
    # train robust FQI
    trn_iters = 0
    max_policy_value = -np.inf
    while trn_iters < args.max_trn_steps:
        # eval. policy and log rewards
        with torch.no_grad():
            avg, std = eval_policy(policy, args.env, 
                                   eval_episodes=args.eval_episodes)
            # log tensorboard
            writer.add_scalar("eval reward", avg, trn_iters)
            writer.add_scalar("eval reward std", std, trn_iters)
            writer.flush()
            if avg > max_policy_value:
                save_policy(policy, save_path)
                if  'FrozenLake' not in args.env and args.video == 'True':
                    generate_gif(policy, args.env, video_path)
                max_policy_value = avg
        # step
        policy.train(data, int(args.eval_freq), batch_size=args.batch_size,
                     writer=writer, log_base=trn_iters)
        # update steps
        trn_iters += args.eval_freq
    writer.close() 
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # epsilon used to generate data
    parser.add_argument('--data_eps', default=0.5, type=float)
    # policy used to generate data
    parser.add_argument('--gendata_pol', default='dqn', type=str)  
    parser.add_argument('--env', default='CartPole-v0', type=str)
    parser.add_argument('--max_trn_steps', default=5e5, type=float)
    parser.add_argument('--eval_freq', default=1e3, type=float)
    parser.add_argument('--eval_episodes', default=10, type=int)
    parser.add_argument('--video', default='False', type=str)
    parser.add_argument('--seed', default=1024, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--data_size', default=1e6, type=int)
    parser.add_argument('--batch_size', default=1e3, type=int)
    
    # epsilon in Adam*
    parser.add_argument('--adam_eps', default=1e-6, type=float)
    # Adam stepsize*
    parser.add_argument('--adam_lr', default=3e-4, type=float)
    # actor lr*
    parser.add_argument('--actor_lr', default=3e-4, type=float)
    # critic lr*
    parser.add_argument('--critic_lr', default=1e-3, type=float)
    parser.add_argument('--rho', default=0.5, type=float)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--tau', default=0.005, type=float)
    parser.add_argument('--lmbda', default=0.75, type=float)                  
    parser.add_argument('--phi', default=0.1, type=float)
    parser.add_argument('--comment', default='', type=str)  
    # use d4rl dataset
    parser.add_argument("--d4rl", default='False', type=str)
    parser.add_argument("--d4rl_v2", default='False', type=str)
    parser.add_argument("--d4rl_expert", default='False', type=str)
    # use mixed dataset
    parser.add_argument("--mixed", default='False', type=str)

    args = parser.parse_args()

    # make folders to dump results
    if not os.path.exists("./logs"):
        os.makedirs("./logs")

    if not os.path.exists("./models"):
        os.makedirs("./models")
    
    if not os.path.exists("./videos"):
        os.makedirs("./videos")

    env = gym.make(args.env)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
        
    # determine action dimension, action limits
    env_action_type = get_action_type(env.action_space)
    if env_action_type == 'continuous':
        action_dim = 1
        max_action = env.action_space.high
        min_action = env.action_space.low
    elif env_action_type == 'discrete':
        action_dim = 1
        max_action = env.action_space.n - 1
        min_action = 0
    elif env_action_type == 'multi_continuous':
        action_dim = env.action_space.shape[0]
        max_action = env.action_space.high
        min_action = env.action_space.low
    elif env_action_type == 'multi_discrete':
        action_dim = env.action_space.shape[0]
        max_action = env.action_space.nvec.max()
        min_action = env.action_space.nvec.min()
    elif env_action_type == 'multi_binary':
        action_dim = env.actoin_space.n
        max_action = 1
        min_action = 0
    else:
        raise NotImplementedError

    # determine state dimensions
    if isinstance(env.observation_space, spaces.Discrete):
        state_dim = 1
        max_state = env.observation_space.n - 1
    else:
        state_dim = env.observation_space.shape[0]
        max_state = np.inf
   
    # check device
    if args.device not in ['cpu', 'cuda', 'cuda:0', 'cuda:1', 'auto']:
        raise NotImplementedError
    
    # check d4rl option and mixed option
    # determine data_path, log_path and save_path
    if args.d4rl == 'False' and args.mixed == 'False':
        data_path = f'offline_data/{args.env}_{args.gendata_pol}_e{args.data_eps}'
        save_path = f'RFQI_{args.env}_rho{args.rho}_dataeps{args.data_eps}_datapol{args.gendata_pol}{args.comment}'
    elif args.d4rl == 'True' and args.mixed == 'False':
        env_subname = args.env[0:args.env.find('-')].lower()
        data_path = f'offline_data/d4rl-{env_subname}'
        save_path = f'RFQI_{args.env}_rho{args.rho}_d4rl'
        if args.d4rl_expert == 'True':
            data_path += '-expert'
            save_path += '_expert'
        else:
            data_path += '-medium'
        if args.d4rl_v2 == 'False':
            data_path += '-v0'
        else:
            data_path += '-v2'
        save_path += args.comment
    elif args.d4rl == 'False' and args.mixed == 'True':
        data_path = f'offline_data/{args.env}_{args.gendata_pol}_mixed_e{args.data_eps}'
        save_path = f'RFQI_mixed_{args.env}_rho{args.rho}_dataeps{args.data_eps}_datapol{args.gendata_pol}{args.comment}'
    else:
        raise NotImplementedError
    paths = dict(data_path=data_path, save_path=save_path)
    
    print("=========================================================")
    print(f'===============Training RFQI on {args.env}==============')
    print(f'{args.env} attributes: max_action={max_action}')
    print(f'                       min_action={min_action}')
    print(f'                       action_dim={action_dim}')
    print(f'                       env action type is {env_action_type}')
    print(f'Training attributes:   using device: {args.device}')
    print(f'                       data path: {data_path}')
    if args.d4rl == 'False':
        print(f'                       using data generated by: {args.gendata_pol}')
    else:
        print(f'                       using d4rl data')
    print(f'                       Adam learning rate: {args.adam_lr}')
    print(f'                       Adam epsilon: {args.adam_eps}')
    print(f'                       actor learning rate: {args.actor_lr}')
    print(f'                       critic learning rate: {args.critic_lr}')
    print("=========================================================")

    train_rfqi(state_dim, action_dim, min_action, max_action, paths,
               env_action_type, args)
