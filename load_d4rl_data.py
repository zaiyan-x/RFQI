import gym
import d4rl
import os
import numpy as np
from data_container import DATA

if not os.path.exists("./offline_data"):
    os.makedirs("./offline_data")

for env_name in ["hopper-medium-v0", "halfcheetah-medium-v0",
                 "hopper-medium-v2", "halfcheetah-medium-v2"]:
    dataset_name = "offline_data/d4rl-" + env_name

    env = gym.make(env_name)
    dataset = env.get_dataset()
    N = dataset['observations'].shape[0]

    dataset['next_observations'] = np.copy(dataset['observations'])
    dataset['next_observations'][0:N - 1, :] = dataset['next_observations'][1:N, :]

    dataset['observations'] = dataset['observations'][~dataset['timeouts'], :]
    dataset['actions'] = dataset['actions'][~dataset['timeouts'], :]
    dataset['next_observations'] = dataset['next_observations'][~dataset['timeouts'], :]
    dataset['rewards'] = dataset['rewards'][~dataset['timeouts']]
    dataset['not_done'] = 1.0 - dataset['terminals'][~dataset['timeouts']].astype(np.float)

    print(len(dataset['observations']),
          len(dataset['actions']),
          len(dataset['next_observations']),
          len(dataset['rewards']),
          len(dataset['not_done']))

    replay_buffer = DATA(env.observation_space.shape[0], env.action_space.shape[0],
                         "cpu", len(dataset['observations']))
    replay_buffer.state = dataset['observations']
    replay_buffer.action = dataset['actions']
    replay_buffer.next_state = dataset['next_observations']
    replay_buffer.reward[:, 0] = dataset['rewards']
    replay_buffer.not_done[:, 0] = dataset['not_done']
    replay_buffer.size = len(dataset['observations'])
    replay_buffer.ptr = replay_buffer.size
    replay_buffer.save(dataset_name)
