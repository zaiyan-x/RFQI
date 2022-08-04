import numpy as np
from stable_baselines3 import PPO, SAC, DQN, TD3
import gym
from gym import spaces
from data_container import DATA
import os
import argparse

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
        
def generate_dataset(env_name, gendata_pol, epsilon, state_dim, action_dim,
                     args, buffer_size=int(1e6), verbose=False):
    # determine trained policy save path and where to save dataset
    if args.mixed == 'True':
        dataset_name = f'./offline_data/{env_name}_{gendata_pol}_mixed_e{epsilon}'
        policy_path = f'./models/{gendata_pol}_mixed_{env_name}'
    else:
        dataset_name = f'./offline_data/{env_name}_{gendata_pol}_e{epsilon}'
        policy_path = f'./models/{gendata_pol}_{env_name}'
    
    if gendata_pol == 'ppo':
        policy = PPO.load(policy_path, device=args.device)
    elif gendata_pol == 'sac':
        policy = SAC.load(policy_path, device=args.device)
    elif gendata_pol == 'dqn':
        policy = DQN.load(policy_path, device=args.device)
    elif gendata_pol == 'td3':
        policy = TD3.load(policy_path, device=args.device)
    else:
        raise NotImplementedError
        
    # prep. environment
    env = gym.make(env_name)
    env_action_type = get_action_type(env.action_space)
        
    data = DATA(state_dim, action_dim, 'cpu', buffer_size)
    states = []
    actions = []
    next_states = []
    rewards = []
    not_dones = []
    
    # set path
    if args.mixed == 'True':
        dataset_name = f'./offline_data/{env_name}_{gendata_pol}_mixed_e{epsilon}'
    else:
        dataset_name = f'./offline_data/{env_name}_{gendata_pol}_e{epsilon}'
    
    # generate dateset
    count = 0
    while count < buffer_size:
        state, done = env.reset(), False
        if verbose:
            print(f'buffer size={buffer_size}======current count={count}')
        while not done:
            if np.random.binomial(n=1, p=epsilon):
                action = env.action_space.sample()
            else: # else we select expert action
                action, _ = policy.predict(state)
                if 'FrozenLake' in env_name:
                    action = int(action)
            next_state, reward, done, _ = env.step(action)
            states.append(state)
            # determine the correct data structure for action
            if env_action_type == 'continuous' or env_action_type == 'discrete':
                action = np.array([action])
            elif env_action_type == 'multi_continuous' or env_action_type == 'multi_discrete' or env_action_type == 'multi_binary':
                action = np.array(action)
            else:
                raise NotImplementedError
                
            if np.random.binomial(n=1, p=0.001):
                print('==================================================')
                print('--------random printing offline data point--------')
                print(f'action: {action}')
                print(f'next_state: {next_state}')
                print(f'not_done: {1.0 - done}')
                print(f'reward: {reward}')
            actions.append(action)
            next_states.append(next_state)
            not_dones.append(np.array([1.0 - done]))
            rewards.append(np.array([reward]))
        
            # check buffer size
            count += 1
            if count >= buffer_size:
                break
            else:    # state transition
                state = next_state 
            
        
    data.state = np.array(states)
    data.state = np.resize(data.state, (buffer_size, state_dim))
    data.action = np.array(actions)
    data.action = np.resize(data.action, (buffer_size, action_dim))
    data.next_state = np.array(next_states)
    data.next_state = np.resize(data.next_state, (buffer_size, state_dim))
    data.reward = np.array(rewards)
    data.not_done = np.array(not_dones)
    data.size = buffer_size
    data.ptr = buffer_size
    data.save(dataset_name)

     
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # OpenAI gym environment name (need to be consistent with the dataset name)
    parser.add_argument("--env", default='CartPole-v1')
    # e-mix (prob. to mix random actions)
    parser.add_argument("--eps", default=0.5, type=float)
    parser.add_argument("--buffer_size", default=1e6, type=float)
    parser.add_argument("--verbose", default='False', type=str)
    parser.add_argument("--device", default='cpu', type=str)
    parser.add_argument("--gendata_pol", default='ppo', type=str)
    # if gendata_pol is trained with mixed traj.
    parser.add_argument("--mixed", default='False', type=str)
    args = parser.parse_args()
    
    if args.verbose == 'False':
        verbose = False
    else:
        verbose = True
        
    # determine dimensions
    env = gym.make(args.env)
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
    
    if isinstance(env.observation_space, spaces.Discrete):
        state_dim = 1
    else:
        state_dim = env.observation_space.shape[0]
    
    # client input sanity check
    if args.device not in ['cpu', 'cuda', 'cuda:0', 'cuda:1', 'auto']:
        raise NotImplementedError
        
    # check path
    if not os.path.exists("./offline_data"):
        os.makedirs("./offline_data")
        
    # check mixed option
    if args.mixed == 'True' or args.mixed == 'False':
        pass
    else:
        raise NotImplementedError
        
    generate_dataset(args.env, args.gendata_pol, args.eps, state_dim, action_dim,
                     args, int(args.buffer_size), verbose)
