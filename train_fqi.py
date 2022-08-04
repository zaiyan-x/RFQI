import argparse
import gym
from gym import spaces
import numpy as np
import os
import torch
from torch.utils.tensorboard import SummaryWriter
import imageio

from fqi import PQL_BCQ
from data_container import DATA

def train_PQL_BCQ(state_dim, action_dim, max_state, min_action, max_action, 
                  paths, args):
    # parse paths
    data_path = paths['data_path']
    log_path = f"./logs/{paths['save_path']}"
    save_path = f"./models/{paths['save_path']}"
    video_path = f"./videos/{paths['save_path']}.gif"
    
    # fix args.automatic_beta type issue
    if args.automatic_beta == 'True':
        automatic_beta = True
    else:
        automatic_beta = False
        
    # tensorboard logger
    writer = SummaryWriter(log_path)
        
    print("=== Start Train ===\n")
    print("Args:\n",args)

    # initialize policy
    policy = PQL_BCQ(state_dim, action_dim, max_state, min_action, max_action, 
                     args.device, args.discount, args.tau, args.lmbda, args.phi,
                     n_action=args.n_action,
                     n_action_execute=args.n_action_execute,
                     backup=args.backup, ql_noise=args.ql_noise,
                     actor_lr=args.actor_lr, beta=args.beta, vmin=args.vmin)

    # load data
    data = DATA(state_dim, action_dim, args.device)
    data.load(data_path, args.data_size)

    # train VAE
    filter_scores = []
    training_iters = 0
    while training_iters < args.max_vae_trn_step:
        vae_loss = policy.train_vae(data, iterations=int(args.eval_freq), batch_size=args.batch_size)
        print(f"Training iterations: {training_iters}. State VAE loss: {vae_loss:.3f}.")
        training_iters += args.eval_freq

    if automatic_beta:  # args.automatic_beta:
        test_loss = policy.test_vae(data, batch_size=100000)
        beta = np.percentile(test_loss, args.beta_percentile)
        policy.beta = beta
        print("Test vae",args.beta_percentile,"percentile:", beta)
    else:
        pass
    
    training_iters = 0
    max_policy_value = -np.inf
    while training_iters < args.max_trn_steps:
        # run policy for 10 episodes and collect avg. and std.
        std, avg = eval_policy(policy, args.env, eval_episodes=10)
        # log tensorboard
        writer.add_scalar("eval reward", avg, training_iters)
        writer.add_scalar("eval reward std", std, training_iters)
        writer.flush()
        # only save model if current policy outperforms the current best
        if avg > max_policy_value:
            save_policy(policy, save_path)
            if  'FrozenLake' not in args.env and args.video == 'True':
                    generate_gif(policy, args.env, video_path)
            max_policy_value = avg

        # train policy for 'eval_freq' steps
        policy.train(data, iterations=int(args.eval_freq), 
                     batch_size=args.batch_size)
                
        training_iters += args.eval_freq # loop
        print(f"Training iterations: {training_iters}")
    writer.close()
    
# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, eval_episodes=10):
    eval_env = gym.make(env_name)

    # Add-ons
    # properly determine action type
    env_action_type = get_action_type(env.action_space)  
    episode_reward = 0.0
    rewards = []
    for _ in range(eval_episodes):
        state, done = eval_env.reset(seed = np.random.randint(10000)), False
        while not done:
            action = policy.select_action(np.array(state))
            if env_action_type == 'discrete':
                action = np.rint(action[0]).astype(int)
            elif env_action_type == 'continuous':
                action = action[0]
            else:
                pass
            state, reward, done, _ = eval_env.step(action)
            episode_reward += reward
        rewards.append(episode_reward)
        episode_reward = 0.0
    reward_std = np.std(rewards)
    avg_reward = np.sum(rewards) / eval_episodes    
    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return reward_std, avg_reward

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
                    [np.array(img) for i, 
                    img in enumerate(images) if i%2==0],
                    fps=29)

def save_policy(policy, save_path):
    policy.actor.save(f'{save_path}_actor')
    policy.critic.save(f'{save_path}_critic')
    policy.vae.save(f'{save_path}_action_vae')
    policy.vae2.save(f'{save_path}_state_vae')

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
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # OpenAI gym environment name (need to be consistent with the dataset name)
    parser.add_argument("--env", default="Hopper-v2", type=str)  
    # policy used to generate data
    parser.add_argument("--gendata_pol", default="ppo", type=str)
    # epsilon used in data generation
    parser.add_argument('--data_eps', default=0.5, type=float)
    # sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--seed", default=1, type=int)  
    # How often (time steps) we evaluate
    parser.add_argument("--eval_freq", default=1e4, type=float)  
    # max time steps to train FQI
    parser.add_argument('--max_trn_steps', default=5e5, type=int)
    parser.add_argument("--max_vae_trn_step", default=2e5, type=int)    
    # number of samples to load into the buffer
    parser.add_argument("--data_size", default=1000000, type=int) 
    # use d4rl dataset
    parser.add_argument("--d4rl", default='False', type=str)
    parser.add_argument("--d4rl_v2", default='False', type=str)
    parser.add_argument("--d4rl_expert", default='False', type=str)
    # use mixed dataset
    parser.add_argument("--mixed", default='False', type=str)
    # extra comment
    parser.add_argument("--comment", default='', type=str)
    # save video
    parser.add_argument("--video", default='False', type=str)

    # mini batch size for networks
    parser.add_argument("--batch_size", default=100, type=int)  
    # discount factor
    parser.add_argument("--discount", default=0.99)  
    # target network update rate
    parser.add_argument("--tau", default=0.005)  
    # weighting for clipped double Q-learning in BCQ
    parser.add_argument("--lmbda", default=0.75)  
    # max perturbation hyper-parameter for BCQ
    parser.add_argument("--phi", default=0.1, type=float)
    # learning rate of actor
    parser.add_argument("--actor_lr", default=1e-3, type=float) 
    # number of sampling action for policy (in backup)
    parser.add_argument("--n_action", default=100, type=int) 
    # number of sampling action for policy (in execution)
    parser.add_argument("--n_action_execute", default=100, type=int) 

    # BCQ-PQL parameter
    # "QL": q learning (Q-max) back up, "AC": actor-critic backup
    parser.add_argument("--backup", type=str, default="QL") 
    # noise of next action in QL
    parser.add_argument("--ql_noise", type=float, default=0.15) 
    # if true, use percentile for b (beta is the b in paper)
    parser.add_argument("--automatic_beta", type=str, default='True') 
    # use x-Percentile as the value of b
    parser.add_argument("--beta_percentile", type=float, default=2.0) 
    # hardcoded b, only effective when automatic_beta = False
    parser.add_argument("--beta", default=-0.4, type=float)  
    # min value of the environment
    # empirically set it to be the min of 1000 random rollout.
    parser.add_argument("--vmin", default=0, type=float) 
    # device to run training
    parser.add_argument("--device", default='cuda', type=str) 

    args = parser.parse_args()

    print("---------------------------------------")
    print(f"Setting: Training PQL-BCQ, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

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

    # Add-ons
    # 1. properly determine action_dim
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
    
    # 2. change from continuous to discrete arguments
    if isinstance(env.observation_space, spaces.Discrete):
        state_dim = 1
        max_state = env.observation_space.n - 1
    else:
        state_dim = env.observation_space.shape[0]
        max_state = None
        
    # 3. check device
    if args.device not in ['cpu', 'cuda', 'cuda:0', 'cuda:1', 'auto']:
        raise NotImplementedError
        
    # check d4rl option and mixed option
    # determine data_path, log_path and save_path
    if args.d4rl == 'False' and args.mixed == 'False':
        data_path = f'offline_data/{args.env}_{args.gendata_pol}_e{args.data_eps}'
        save_path = f'FQI_{args.env}_dataeps{args.data_eps}_datapol{args.gendata_pol}{args.comment}'
    elif args.d4rl == 'True' and args.mixed == 'False':
        env_subname = args.env[0:args.env.find('-')].lower()
        data_path = f'offline_data/d4rl-{env_subname}'
        save_path = f'FQI_{args.env}_d4rl'
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
        save_path = f'FQI_mixed_{args.env}_dataeps{args.data_eps}_datapol{args.gendata_pol}{args.comment}'
    else:
        raise NotImplementedError
    paths = dict(data_path=data_path, save_path=save_path)     
    
    print("=========================================================")
    print(f'===============Training FQI on {args.env}==============')
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
    print(f'                       actor learning rate: {args.actor_lr}')
    print(f'                       extra comment: {args.comment}')
    print("=========================================================")

    train_PQL_BCQ(state_dim, action_dim, max_state, min_action, max_action,
                  paths, args)
