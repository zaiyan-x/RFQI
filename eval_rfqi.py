import argparse
import gym
from gym import spaces
import numpy as np
import os
import torch
import imageio

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
        
def predict(policy, state, env_action_type):
    '''
    PQL (RFQI) version of predict.
    '''
    action = policy.select_action(np.array(state))
    if env_action_type == 'discrete':
        action = np.rint(action[0]).astype(int)
    elif env_action_type == 'continuous':
        action = action[0]
    elif env_action_type == 'multi_continuous':
        return action
    else:
        raise NotImplementedError
    return action

def load_policy(policy, load_path, device):
    policy.actor.load(f'{load_path}_actor', device=device)
    policy.critic.load(f'{load_path}_critic', device=device)
    policy.vae.load(f'{load_path}_vae', device=device)
    return policy

def save_evals(save_path, setting, avgs, stds):
    np.save(f'{save_path}_{setting}_avgs', avgs)
    np.save(f'{save_path}_{setting}_stds', stds)


def eval_rfqi(state_dim, action_dim, min_action, max_action, hard, paths,
              perturbed_env, args):
    '''
    Evaluate RFQI on perturbed environments.
    '''
    # parse paths
    load_path = f"./models/{paths['load_path']}"
    save_path = f"./perturbed_results/{paths['save_path']}"
    video_path = f"./perturbed_videos/{paths['save_path']}"
    if not os.path.exists(video_path):
        os.makedirs(video_path)
    
    # evaluation attributes
    eval_episodes = args.eval_episodes
    
    # make environment and identify settings to be perturbed
    env = gym.make(perturbed_env)
    # get the action type
    env_action_type = get_action_type(env.action_space)
    # Initialize policy
    policy = RFQI(state_dim, action_dim, min_action, max_action, args.device,
                  env_action_type=env_action_type, gamma=args.gamma, tau=args.tau,
                  lmbda=args.lmbda, phi=args.phi)
    policy = load_policy(policy, load_path, args.device)
    ######################################################
    ####--------------CartPolePerturbed-v0-----------#####
    ######################################################
    if args.env == 'CartPole-v0':
        ps_fm = np.arange(-0.8, 4.0, 0.1)
        ps_g = np.arange(-3.0, 3.0, 0.1)
        ps_len = np.arange(-0.8, 4.0, 0.1)
        settings = ['force_mag', 'gravity', 'length']
        # perturb 'force_mag'
        setting = 'force_mag'
        avgs = []
        stds = []
        for p in ps_fm:
            env.reset()
            force_mag = env.force_mag * (1 + p)
            rewards = []
            for _ in range(eval_episodes):
                # complete random environment for each episode
                if hard:
                    state, done = env.reset(force_mag=force_mag, 
                                            seed = np.random.randint(10000),
                                            init_angle_mag=0.2), False
                else: 
                    state, done = env.reset(force_mag=force_mag,
                                            init_angle_mag=0.2), False
                episode_reward = 0.0
                while not done:
                    action = predict(policy, state, env_action_type)
                    state, reward, done, _ = env.step(action)
                    episode_reward += reward
                # episode done
                rewards.append(episode_reward)
            # episodes for current p are done
            avg_reward = np.sum(rewards) / eval_episodes
            print("---------------------------------------")
            print(f' force_mag with p {p}')
            print(f" over {eval_episodes} episodes: {avg_reward:.3f}")
            print("---------------------------------------")
            avgs.append(avg_reward)
            stds.append(np.std(rewards))
        # all p's are done
        save_evals(save_path, setting, avgs, stds)
        
        # perturb 'gravity'
        setting = 'gravity'
        avgs = []
        stds = []
        for p in ps_g:
            env.reset()
            gravity = env.gravity * (1 + p)
            rewards = []
            for _ in range(eval_episodes):
                # complete random environment for each episode
                if hard:
                    state, done = env.reset(gravity=gravity, 
                                            seed = np.random.randint(10000),
                                            init_angle_mag=0.2), False
                else:
                    state, done = env.reset(gravity=gravity,
                                            init_angle_mag=0.2), False
                episode_reward = 0.0
                while not done:
                    action = predict(policy, state, env_action_type)
                    state, reward, done, _ = env.step(action)
                    episode_reward += reward
                # episode done
                rewards.append(episode_reward)
                
            # episodes for current p are done
            avg_reward = np.sum(rewards) / eval_episodes
            print("---------------------------------------")
            print(f' gravity with p {p}')
            print(f" over {eval_episodes} episodes: {avg_reward:.3f}")
            print("---------------------------------------")
            avgs.append(avg_reward)
            stds.append(np.std(rewards))
        # all p's are done
        save_evals(save_path, setting, avgs, stds)
        
        # perturb 'length'
        setting = 'length'
        avgs = []
        stds = []
        ps = np.arange(-0.8, 4.0, 0.1)
        for p in ps_len:
            env.reset()
            length = env.length * (1 + p)
            rewards = []
            for _ in range(eval_episodes):
                # complete random environment for each episode
                if hard:
                    state, done = env.reset(length=length, 
                                            seed = np.random.randint(10000),
                                            init_angle_mag=0.2), False
                else: 
                    state, done = env.reset(gravity=gravity,
                                            init_angle_mag=0.2), False
                episode_reward = 0.0
                while not done:
                    action = predict(policy, state, env_action_type)
                    state, reward, done, _ = env.step(action)
                    episode_reward += reward
                # episode done
                rewards.append(episode_reward)
                
            # episodes for current p are done
            avg_reward = np.sum(rewards) / eval_episodes
            print("---------------------------------------")
            print(f' length with p {p}')
            print(f" over {eval_episodes} episodes: {avg_reward:.3f}")
            print("---------------------------------------")
            avgs.append(avg_reward)
            stds.append(np.std(rewards))
        # all p's are done
        save_evals(save_path, setting, avgs, stds)
        
        # perturb actions
        setting = 'action'
        avgs = []
        stds = []
        es = np.arange(0,1.1,0.1)
        for e in es:
            env.reset()
            rewards = []
            for _ in range(eval_episodes):
                # complete random environment for each episode
                if hard:
                    state, done = env.reset(seed=np.random.randint(10000)), False
                else:
                    state, done = env.reset(), False
                episode_reward = 0.0
                while not done:
                    if np.random.binomial(n=1, p=e):
                        action = env.action_space.sample()
                    else: # else we use policy
                        action = predict(policy, state, env_action_type)
                    state, reward, done, _ = env.step(action)
                    episode_reward += reward
                # episode done
                rewards.append(episode_reward)
                
            # episodes for current p are done
            avg_reward = np.sum(rewards) / eval_episodes
            print("---------------------------------------")
            print(f' action with e {e}')
            print(f" over {eval_episodes} episodes: {avg_reward:.3f}")
            print("---------------------------------------")
            avgs.append(avg_reward)
            stds.append(np.std(rewards))
        # all p's are done
        save_evals(save_path, setting, avgs, stds)
    ######################################################
    ####-----------LunarLanderPerturbed-v2-----------#####
    ####------LunarLanderContinuousPerturbed-v2------#####
    ######################################################
    if (args.env == 'LunarLanderContinuous-v2' or
        args.env == 'LunarLander-v2'):
        settings = ['engine_power', 'gravity', 'wind', 'initial_random',
                    'initial_x']
        INITIAL_RANDOM = 1500.0
        ps_ep = np.arange(-0.9, 1.1, 0.1)
        ps_g = np.arange(-0.99, 0.0, 0.1)
        ps_w = np.arange(-0.99, 0.3, 0.1)
        ps_ir = np.arange(0.0, 0.55, 0.05)
        ps_ix = np.arange(-1.0, 1.0, 0.1)
        # perturb 'engine_power'
        setting = 'engine_power'
        avgs = []
        stds = []
        for p in ps_ep:
            env.reset()
            main_engine_power = env.main_engine_power * (1 + p)
            side_engine_power = env.side_engine_power * (1 + p)
            # eval
            rewards = []
            for i in range(eval_episodes):
                if i == 0: # save video
                    images = []
                    img = env.render(mode='rgb_array')
                    curr_videopath = f'{video_path}/{setting}_{p:.2f}.gif'
                # reset env
                if args.env == 'LunarLanderContinuous-v2':
                    if hard:
                        state = env.reset(seed=np.random.randint(10000),
                                          enable_wind=True,
                                          initial_random=INITIAL_RANDOM,
                                          main_engine_power=main_engine_power, 
                                          side_engine_power=side_engine_power)
                        done = False
                    else:
                        state = env.reset(main_engine_power=main_engine_power, 
                                          enable_wind=True,
                                          initial_random=INITIAL_RANDOM,
                                          side_engine_power=side_engine_power)

                        done = False
                else: # 'LunarLander-v2'
                    if hard:
                        state = env.reset(seed=np.random.randint(10000),
                                          main_engine_power=main_engine_power, 
                                          side_engine_power=side_engine_power)
                        done = False
                    else:
                        state = env.reset(main_engine_power=main_engine_power, 
                                          side_engine_power=side_engine_power)

                        done = False
                episode_reward = 0.0
                while not done:
                    if i == 0: # save video
                        images.append(img)
                    
                    action = predict(policy, state, env_action_type)
                    state, reward, done, _ = env.step(action)
                    episode_reward += reward
                    
                    if i == 0: # save video
                        img = env.render(mode='rgb_array')
                # episode done
                rewards.append(episode_reward)
                if i == 0: # save video
                    imageio.mimsave(curr_videopath, 
                                    [np.array(img) for i, 
                                    img in enumerate(images) if i%2==0],
                                    fps=29)
            # episodes for current p are done
            avg_reward = np.sum(rewards) / eval_episodes
            print("---------------------------------------")
            print(f' engine_power with ep {p}')
            print(f" over {eval_episodes} episodes: {avg_reward:.3f}")
            print("---------------------------------------")
            avgs.append(avg_reward)
            stds.append(np.std(rewards))
        # all p's are done
        save_evals(save_path, setting, avgs, stds)
        
        # perturb 'gravity'
        setting = 'gravity'
        avgs = []
        stds = []
        for p in ps_g:
            env.reset()
            gravity = env.gravity * (1 + p)
            rewards = []
            for i in range(eval_episodes):
                if i == 0: # save video
                    images = []
                    img = env.render(mode='rgb_array')
                    curr_videopath = f'{video_path}/{setting}_{p:.2f}.gif'
                    
                if args.env == 'LunarLanderContinuous-v2':
                    if hard:
                        state, done = env.reset(seed=np.random.randint(10000),
                                                enable_wind=True,
                                                initial_random=INITIAL_RANDOM,
                                                gravity=gravity), False
                    else:
                        state, done = env.reset(enable_wind=True,
                                                initial_random=INITIAL_RANDOM,
                                                gravity=gravity), False
                else: # 'LunarLander-v2'
                    if hard:
                        state, done = env.reset(seed=np.random.randint(10000),
                                                gravity=gravity), False
                    else:
                        state, done = env.reset(gravity=gravity), False
                
                
                episode_reward = 0.0
                while not done:
                    if i == 0: # save video
                        images.append(img)
                    action = predict(policy, state, env_action_type)
                    state, reward, done, _ = env.step(action)
                    episode_reward += reward
                    if i == 0: # save video
                        img = env.render(mode='rgb_array')
                # episode done
                rewards.append(episode_reward)
                if i == 0: # save video
                    imageio.mimsave(curr_videopath, 
                                    [np.array(img) for i, 
                                    img in enumerate(images) if i%2==0],
                                    fps=29)
            # episodes for current p are done
            avg_reward = np.sum(rewards) / eval_episodes
            print("---------------------------------------")
            print(f' gravity with p {p}')
            print(f" over {eval_episodes} episodes: {avg_reward:.3f}")
            print("---------------------------------------")
            avgs.append(avg_reward)
            stds.append(np.std(rewards))
        # all p's are done
        save_evals(save_path, setting, avgs, stds)
        
        # perturb 'wind'
        setting = 'wind'
        avgs = []
        stds = []
        for p in ps_w:
            env.reset()
            wind_power = env.wind_power * (1 + p)
            rewards = []
            for i in range(eval_episodes):
                if i == 0: # save video
                    images = []
                    img = env.render(mode='rgb_array')
                    curr_videopath = f'{video_path}/{setting}_{p:.2f}.gif'
                if args.env == 'LunarLanderContinuous-v2':
                    if hard:
                        state, done = env.reset(seed=np.random.randint(10000),
                                                enable_wind=True,
                                                initial_random=INITIAL_RANDOM,
                                                wind_power=wind_power), False
                    else:
                        state, done = env.reset(enable_wind=True,
                                                initial_random=INITIAL_RANDOM,
                                                wind_power=wind_power), False
                else: # 'LunarLander-v2'
                    if hard:
                        state, done = env.reset(seed=np.random.randint(10000),
                                                enable_wind=True,
                                                wind_power=wind_power), False
                    else:
                        state, done = env.reset(enable_wind=True,
                                                wind_power=wind_power), False
                
                
                episode_reward = 0.0
                while not done:
                    if i == 0: # save video
                        images.append(img)
                        
                    action = predict(policy, state, env_action_type)
                    state, reward, done, _ = env.step(action)
                    episode_reward += reward
                    
                    if i == 0: # save video
                        img = env.render(mode='rgb_array')
                # episode done
                rewards.append(episode_reward)
                if i == 0: # save video
                    imageio.mimsave(curr_videopath, 
                                    [np.array(img) for i, 
                                    img in enumerate(images) if i%2==0],
                                    fps=29)
            # episodes for current p are done
            avg_reward = np.sum(rewards) / eval_episodes
            print("---------------------------------------")
            print(f' wind with p {p}')
            print(f" over {eval_episodes} episodes: {avg_reward:.3f}")
            print("---------------------------------------")
            avgs.append(avg_reward)
            stds.append(np.std(rewards))
        # all p's are done
        save_evals(save_path, setting, avgs, stds)
        
        # perturb 'initial_random'
        setting = 'initial_random'
        avgs = []
        stds = []
        for p in ps_ir:
            env.reset()
            initial_random = env.initial_random * (1 + p)
            rewards = []
            for i in range(eval_episodes):
                if i == 0: # save video
                    images = []
                    img = env.render(mode='rgb_array')
                    curr_videopath = f'{video_path}/{setting}_{p:.2f}.gif'
                    
                if args.env == 'LunarLanderContinuous-v2':
                    if hard:
                        state = env.reset(seed=np.random.randint(10000),
                                          enable_wind=True,
                                          initial_random=initial_random)
                        done = False
                    else:
                        state = env.reset(enable_wind=True,
                                          initial_random=initial_random)
                else: # 'LunarLander-v2'
                    if hard:
                        state = env.reset(seed=np.random.randint(10000),
                                          initial_random=initial_random)
                        done = False
                    else:
                        state = env.reset(initial_random=initial_random)    
                
                    done = False
                episode_reward = 0.0
                while not done:
                    if i == 0: # save video
                        images.append(img)
                        
                    action = predict(policy, state, env_action_type)
                    state, reward, done, _ = env.step(action)
                    episode_reward += reward
                    
                    if i == 0: # save video
                        img = env.render(mode='rgb_array')
                # episode done
                rewards.append(episode_reward)
                if i == 0: # save video
                    imageio.mimsave(curr_videopath, 
                                    [np.array(img) for i, 
                                    img in enumerate(images) if i%2==0],
                                    fps=29)
            # episodes for current p are done
            avg_reward = np.sum(rewards) / eval_episodes
            print("---------------------------------------")
            print(f' initial random with p {p}')
            print(f" over {eval_episodes} episodes: {avg_reward:.3f}")
            print("---------------------------------------")
            avgs.append(avg_reward)
            stds.append(np.std(rewards))
        # all p's are done
        save_evals(save_path, setting, avgs, stds)
        
        # perturb 'initial_x'
        setting = 'initial_x'
        avgs = []
        stds = []
        for p in ps_ix:
            env.reset()
            initial_x = env.initial_x * (1 + p)
            rewards = []
            for i in range(eval_episodes):
                if i == 0: # save video
                    images = []
                    img = env.render(mode='rgb_array')
                    curr_videopath = f'{video_path}/{setting}_{p:.2f}.gif'
                    
                if args.env == 'LunarLanderContinuous-v2':
                    if hard:
                        state = env.reset(seed=np.random.randint(10000),
                                          enable_wind=True,
                                          initial_random=INITIAL_RANDOM,
                                          initial_x=initial_x)
                        done = False
                    else:
                        state = env.reset(enable_wind=True,
                                          initial_random=INITIAL_RANDOM,
                                          initial_x=initial_x)
                        done = False
                else: # 'LunarLander-v2'
                    if hard:
                        state = env.reset(seed=np.random.randint(10000),
                                          initial_x=initial_x)
                        done = False
                    else:
                        state = env.reset(initial_x=initial_x)
                        done = False
                
                episode_reward = 0.0
                while not done:
                    if i == 0: # save video
                        images.append(img)
                        
                    action = predict(policy, state, env_action_type)
                    state, reward, done, _ = env.step(action)
                    episode_reward += reward
                    
                    if i == 0: # save video
                        img = env.render(mode='rgb_array')
                # episode done
                rewards.append(episode_reward)
                if i == 0: # save video
                    imageio.mimsave(curr_videopath, 
                                    [np.array(img) for i, 
                                    img in enumerate(images) if i%2==0],
                                    fps=29)
            # episodes for current p are done
            avg_reward = np.sum(rewards) / eval_episodes
            print("---------------------------------------")
            print(f' initial x with p {p}')
            print(f" over {eval_episodes} episodes: {avg_reward:.3f}")
            print("---------------------------------------")
            avgs.append(avg_reward)
            stds.append(np.std(rewards))
        # all p's are done
        save_evals(save_path, setting, avgs, stds)
        
        # perturb actions
        setting = 'action'
        avgs = []
        stds = []
        es = np.arange(0,1.1,0.1)
        for e in es:
            env.reset()
            rewards = []
            for i in range(eval_episodes):
                if i == 0: # save video
                    images = []
                    img = env.render(mode='rgb_array')
                    curr_videopath = f'{video_path}/{setting}_{e:.2f}.gif'
                    
                # reset env    
                if args.env == 'LunarLanderContinuous-v2':
                    state, done = env.reset(seed=np.random.randint(10000),
                                            initial_random=INITIAL_RANDOM,
                                            enable_wind=True), False
                else:
                    state, done = env.reset(seed=np.random.randint(10000)), False
                    
                # eval.   
                episode_reward = 0.0
                while not done:
                    if i == 0: # save video
                        images.append(img)
                        
                    if np.random.binomial(n=1, p=e):
                        action = env.action_space.sample()
                    else: # else we use policy
                        action = predict(policy, state, env_action_type)
                    state, reward, done, _ = env.step(action)
                    episode_reward += reward
                    
                    if i == 0: # save video
                        img = env.render(mode='rgb_array')
                # episode done
                rewards.append(episode_reward)
                if i == 0: # save video
                    imageio.mimsave(curr_videopath, 
                                    [np.array(img) for i, 
                                    img in enumerate(images) if i%2==0],
                                    fps=29)
            # episodes for current p are done
            avg_reward = np.sum(rewards) / eval_episodes
            print("---------------------------------------")
            print(f' action with e {e}')
            print(f" over {eval_episodes} episodes: {avg_reward:.3f}")
            print("---------------------------------------")
            avgs.append(avg_reward)
            stds.append(np.std(rewards))
        # all p's are done
        save_evals(save_path, setting, avgs, stds)         

    ######################################################
    ####-------------HopperPerturbed-v3--------------#####
    ######################################################
    if args.env == 'Hopper-v3':
        settings = ['gravity', 'thigh_joint_stiffness', 'leg_joint_stiffness',
                    'foot_joint_stiffness', 'actuator_ctrlrange']
        springref = 2.0
        ps_g = np.arange(-0.5, 0.0, 0.05)
        xs_thijntstif = np.arange(0.0, 32.5, 2.5)
        xs_legjntstif = np.arange(0.0, 60.0, 5.0)
        xs_ftjntstif = np.arange(0.0, 25.0, 2.5)
        xs_act = np.arange(0.85, 1.025, 0.025)
        ps_damp = np.arange(0.0, 1.1, 0.1)
        xs_fric = [0.0, 1.0, 2.0, 3.0, 4.0]
        es = np.arange(0,0.45,0.05)

        
        # perturb 'gravity'
        setting = 'gravity'
        avgs = []
        stds = []
        for p in ps_g:
            env.reset()
            gravity = env.gravity * (1 + p)
            rewards = []
            for i in range(eval_episodes):
                if i == 0: # save video
                    images = []
                    img = env.render(mode='rgb_array')
                    curr_videopath = f'{video_path}/{setting}_{p:.2f}.gif'
                if hard:
                    state, done = env.reset(seed=np.random.randint(10000),
                                            gravity=gravity), False
                else:
                    state, done = env.reset(gravity=gravity), False
                episode_reward = 0.0
                while not done:
                    if i == 0: # save video
                        images.append(img)
                    
                    action = predict(policy, state, env_action_type)
                    state, reward, done, _ = env.step(action)
                    episode_reward += reward
                    if i == 0: # save video
                        img = env.render(mode='rgb_array')
                # episode done
                rewards.append(episode_reward)
                if i == 0: # save video
                    imageio.mimsave(curr_videopath, 
                                    [np.array(img) for i, 
                                    img in enumerate(images) if i%2==0],
                                    fps=29)
                
            # episodes for current p are done
            avg_reward = np.sum(rewards) / eval_episodes
            print("---------------------------------------")
            print(f' gravity with p {p}')
            print(f" over {eval_episodes} episodes: {avg_reward:.3f}")
            print("---------------------------------------")
            avgs.append(avg_reward)
            stds.append(np.std(rewards))
        # all p's are done
        save_evals(save_path, setting, avgs, stds)
        
        # perturb 'thigh_joint_stiffness'
        setting = 'thigh_joint_stiffness'
        avgs = []
        stds = []
        for x in xs_thijntstif:
            rewards = []
            for i in range(eval_episodes):
                if i == 0: # save video
                    images = []
                    img = env.render(mode='rgb_array')
                    curr_videopath = f'{video_path}/{setting}_{x:.2f}.gif'
                if hard:
                    state, done = env.reset(seed=np.random.randint(10000),
                                            springref=springref,
                                            thigh_joint_stiffness=x), False
                else:
                    state, done = env.reset(springref=springref,
                                            thigh_joint_stiffness=x), False
                episode_reward = 0.0
                while not done:
                    if i == 0: # save video
                        images.append(img)
                        
                    action = predict(policy, state, env_action_type)
                    state, reward, done, _ = env.step(action)
                    episode_reward += reward
                    
                    if i == 0: # save video
                        img = env.render(mode='rgb_array')
                # episode done
                rewards.append(episode_reward)
                if i == 0: # save video
                    imageio.mimsave(curr_videopath, 
                                    [np.array(img) for i, 
                                    img in enumerate(images) if i%2==0],
                                    fps=29)      
            # episodes for current p are done
            avg_reward = np.sum(rewards) / eval_episodes
            print("---------------------------------------")
            print(f' thigh joint stiffness with x {x}')
            print(f" over {eval_episodes} episodes: {avg_reward:.3f}")
            print("---------------------------------------")
            avgs.append(avg_reward)
            stds.append(np.std(rewards))
        # all p's are done
        save_evals(save_path, setting, avgs, stds)
        
        # perturb 'leg_joint_stiffness'
        setting = 'leg_joint_stiffness'
        avgs = []
        stds = []
        for x in xs_legjntstif:
            rewards = []
            for i in range(eval_episodes):
                if i == 0: # save video
                    images = []
                    img = env.render(mode='rgb_array')
                    curr_videopath = f'{video_path}/{setting}_{x:.2f}.gif'
                if hard:
                    state, done = env.reset(seed=np.random.randint(10000),
                                            springref=springref,
                                            leg_joint_stiffness=x), False
                else:
                    state, done = env.reset(springref=springref,
                                            leg_joint_stiffness=x), False
                episode_reward = 0.0
                while not done:
                    if i == 0: # save video
                        images.append(img)
                        
                    action = predict(policy, state, env_action_type)
                    state, reward, done, _ = env.step(action)
                    episode_reward += reward
                    
                    if i == 0: # save video
                        img = env.render(mode='rgb_array')
                # episode done
                rewards.append(episode_reward)
                if i == 0: # save video
                    imageio.mimsave(curr_videopath, 
                                    [np.array(img) for i, 
                                    img in enumerate(images) if i%2==0],
                                    fps=29)      
            # episodes for current p are done
            avg_reward = np.sum(rewards) / eval_episodes
            print("---------------------------------------")
            print(f' leg joint stiffness with x {x}')
            print(f" over {eval_episodes} episodes: {avg_reward:.3f}")
            print("---------------------------------------")
            avgs.append(avg_reward)
            stds.append(np.std(rewards))
        # all p's are done
        save_evals(save_path, setting, avgs, stds)
        
        
        # perturb 'foot_joint_stiffness'
        setting = 'foot_joint_stiffness'
        avgs = []
        stds = []
        for x in xs_ftjntstif:
            rewards = []
            for i in range(eval_episodes):
                if i == 0: # save video
                    images = []
                    img = env.render(mode='rgb_array')
                    curr_videopath = f'{video_path}/{setting}_{x:.2f}.gif'
                if hard:
                    state, done = env.reset(seed=np.random.randint(10000),
                                            springref=springref,
                                            foot_joint_stiffness=x), False
                else:
                    state, done = env.reset(springref=springref,
                                            foot_joint_stiffness=x), False
                episode_reward = 0.0
                while not done:
                    if i == 0: # save video
                        images.append(img)
                        
                    action = predict(policy, state, env_action_type)
                    state, reward, done, _ = env.step(action)
                    episode_reward += reward
                    
                    if i == 0: # save video
                        img = env.render(mode='rgb_array')
                # episode done
                rewards.append(episode_reward)
                if i == 0: # save video
                    imageio.mimsave(curr_videopath, 
                                    [np.array(img) for i, 
                                    img in enumerate(images) if i%2==0],
                                    fps=29)      
            # episodes for current p are done
            avg_reward = np.sum(rewards) / eval_episodes
            print("---------------------------------------")
            print(f' foot joint stiffness with x {x}')
            print(f" over {eval_episodes} episodes: {avg_reward:.3f}")
            print("---------------------------------------")
            avgs.append(avg_reward)
            stds.append(np.std(rewards))
        # all p's are done
        save_evals(save_path, setting, avgs, stds)
        
        # perturb actuator control range
        setting = 'actuator_ctrlrange'
        avgs = []
        stds = []
        for x in xs_act:
            rewards = []
            for i in range(eval_episodes):
                if i == 0: # save video
                    images = []
                    img = env.render(mode='rgb_array')
                    curr_videopath = f'{video_path}/{setting}_{x:.2f}.gif'
                if hard:
                    state, done = env.reset(seed=np.random.randint(10000),
                                            actuator_ctrlrange=(-x,x)), False
                else:
                    state, done = env.reset(actuator_ctrlrange=(-x,x)), False
                episode_reward = 0.0
                while not done:
                    if i == 0: # save video
                        images.append(img)
                        
                    action = predict(policy, state, env_action_type)
                    state, reward, done, _ = env.step(action)
                    episode_reward += reward
                    
                    if i == 0: # save video
                        img = env.render(mode='rgb_array')
                # episode done
                rewards.append(episode_reward)
                if i == 0: # save video
                    imageio.mimsave(curr_videopath, 
                                    [np.array(img) for i, 
                                    img in enumerate(images) if i%2==0],
                                    fps=29)      
            # episodes for current p are done
            avg_reward = np.sum(rewards) / eval_episodes
            print("---------------------------------------")
            print(f' actuator ctrlrange is [-{x}, {x}]')
            print(f" over {eval_episodes} episodes: {avg_reward:.3f}")
            print("---------------------------------------")
            avgs.append(avg_reward)
            stds.append(np.std(rewards))
        # all p's are done
        save_evals(save_path, setting, avgs, stds)        
       
        # perturb 'joint_damping'
        setting = 'joint_damping'
        avgs = []
        stds = []
        for p in ps_damp:
            env.reset()
            rewards = []
            for i in range(eval_episodes):
                if i == 0: # save video
                    images = []
                    img = env.render(mode='rgb_array')
                    curr_videopath = f'{video_path}/{setting}_{p:.2f}.gif'
                if hard:
                    state, done = env.reset(seed=np.random.randint(10000),
                                            joint_damping_p=p), False
                else:
                    state, done = env.reset(joint_damping_p=p), False
                episode_reward = 0.0
                while not done:
                    if i == 0: # save video
                        images.append(img)
                    
                    action = predict(policy, state, env_action_type)
                    state, reward, done, _ = env.step(action)
                    episode_reward += reward
                    if i == 0: # save video
                        img = env.render(mode='rgb_array')
                # episode done
                rewards.append(episode_reward)
                if i == 0: # save video
                    imageio.mimsave(curr_videopath, 
                                    [np.array(img) for i, 
                                    img in enumerate(images) if i%2==0],
                                    fps=29)
                
            # episodes for current p are done
            avg_reward = np.sum(rewards) / eval_episodes
            print("---------------------------------------")
            print(f' joint damping with p {p}')
            print(f" over {eval_episodes} episodes: {avg_reward:.3f}")
            print("---------------------------------------")
            avgs.append(avg_reward)
            stds.append(np.std(rewards))
        # all p's are done
        save_evals(save_path, setting, avgs, stds)
        
        # perturb 'joint_frictionloss'
        setting = 'joint_frictionloss'
        avgs = []
        stds = []
        for x in xs_fric:
            rewards = []
            for i in range(eval_episodes):
                if i == 0: # save video
                    images = []
                    img = env.render(mode='rgb_array')
                    curr_videopath = f'{video_path}/{setting}_{x:.2f}.gif'
                if hard:
                    state, done = env.reset(seed=np.random.randint(10000),
                                            joint_frictionloss=x), False
                else:
                    state, done = env.reset(joint_frictionloss=x), False
                episode_reward = 0.0
                while not done:
                    if i == 0: # save video
                        images.append(img)
                        
                    action = predict(policy, state, env_action_type)
                    state, reward, done, _ = env.step(action)
                    episode_reward += reward
                    
                    if i == 0: # save video
                        img = env.render(mode='rgb_array')
                # episode done
                rewards.append(episode_reward)
                if i == 0: # save video
                    imageio.mimsave(curr_videopath, 
                                    [np.array(img) for i, 
                                    img in enumerate(images) if i%2==0],
                                    fps=29)      
            # episodes for current p are done
            avg_reward = np.sum(rewards) / eval_episodes
            print("---------------------------------------")
            print(f'joint frictionloss with x {x}')
            print(f" over {eval_episodes} episodes: {avg_reward:.3f}")
            print("---------------------------------------")
            avgs.append(avg_reward)
            stds.append(np.std(rewards))
        # all p's are done
        save_evals(save_path, setting, avgs, stds)
        
        # perturb actions
        setting = 'action'
        avgs = []
        stds = []
        for e in es:
            env.reset()
            rewards = []
            for i in range(eval_episodes):
                if i == 0: # save video
                    images = []
                    img = env.render(mode='rgb_array')
                    curr_videopath = f'{video_path}/{setting}_{e:.2f}.gif'
                state, done = env.reset(seed=np.random.randint(100000)), False
                episode_reward = 0.0
                while not done:
                    if i == 0: # save video
                        images.append(img)
                        
                    if np.random.binomial(n=1, p=e):
                        action = env.action_space.sample()
                    else: # else we use policy
                        action = predict(policy, state, env_action_type)
                    state, reward, done, _ = env.step(action)
                    episode_reward += reward
                    
                    if i == 0: # save video
                        img = env.render(mode='rgb_array')
                # episode done
                rewards.append(episode_reward)
                if i == 0: # save video
                    imageio.mimsave(curr_videopath, 
                                    [np.array(img) for i, 
                                    img in enumerate(images) if i%2==0],
                                    fps=29)
            # episodes for current p are done
            avg_reward = np.sum(rewards) / eval_episodes
            print("---------------------------------------")
            print(f' action with e {e}')
            print(f" over {eval_episodes} episodes: {avg_reward:.3f}")
            print("---------------------------------------")
            avgs.append(avg_reward)
            stds.append(np.std(rewards))
        # all p's are done
        save_evals(save_path, setting, avgs, stds)          

    ######################################################
    ####-----------HalfCheetahPerturbed-v3-----------#####
    ######################################################
    if args.env == 'HalfCheetah-v3':
        settings = ['gravity', 'back_joint_stiffness', 'front_joint_stiffness',
                    'front_actuator_ctrlrange', 'back_actuator_ctrlrange',
                    'joint_damping', 'joint_frictionloss']
        # for d4rl-medium
        ps_g = np.arange(-0.99, 0.0, 0.1)
        ps_bjntstif = np.arange(0.0, 0.80, 0.05)
        ps_fjntstif = np.arange(0.0, 0.80, 0.05)
        ps_damp = np.arange(-0.5, 0.55, 0.05)
        xs_fact = np.arange(0.3, 1.1, 0.1)
        xs_bact = np.arange(0.3, 1.1, 0.1)
        xs_fric = [0.0, 1.0, 2.0, 3.0, 4.0]
        es = np.arange(0, 1.1, 0.1)
        
        # perturb 'gravity'
        setting = 'gravity'
        avgs = []
        stds = []
        for p in ps_g:
            env.reset()
            gravity = env.gravity * (1 + p)
            rewards = []
            for i in range(eval_episodes):
                if i == 0: # save video
                    images = []
                    img = env.render(mode='rgb_array')
                    curr_videopath = f'{video_path}/{setting}_{p:.2f}.gif'
                if hard:
                    state, done = env.reset(seed=np.random.randint(10000),
                                            gravity=gravity), False
                else:
                    state, done = env.reset(gravity=gravity), False
                episode_reward = 0.0
                while not done:
                    if i == 0: # save video
                        images.append(img)
                    action = predict(policy, state, env_action_type)
                    state, reward, done, _ = env.step(action)
                    episode_reward += reward
                    if i == 0: # save video
                        img = env.render(mode='rgb_array')
                # episode done
                rewards.append(episode_reward)
                if i == 0: # save video
                    imageio.mimsave(curr_videopath, 
                                    [np.array(img) for i, 
                                    img in enumerate(images) if i%2==0],
                                    fps=29)
                
            # episodes for current p are done
            avg_reward = np.sum(rewards) / eval_episodes
            print("---------------------------------------")
            print(f' gravity with p {p} and gravity={gravity}')
            print(f" over {eval_episodes} episodes: {avg_reward:.3f}")
            print("---------------------------------------")
            avgs.append(avg_reward)
            stds.append(np.std(rewards))
        # all p's are done
        save_evals(save_path, setting, avgs, stds)
        
        # perturb 'back_joint_stiffness'
        setting = 'back_joint_stiffness'
        avgs = []
        stds = []
        for p in ps_bjntstif:
            rewards = []
            env.reset()
            bthigh_joint_stiffness = env.bthigh_joint_stiffness * (1 + p)
            bshin_joint_stiffness = env.bshin_joint_stiffness * (1 + p)
            bfoot_joint_stiffness = env.bfoot_joint_stiffness * (1 + p)
            for i in range(eval_episodes):
                if i == 0: # save video
                    images = []
                    img = env.render(mode='rgb_array')
                    curr_videopath = f'{video_path}/{setting}_{p:.2f}.gif'
                if hard:
                    state = env.reset(seed=np.random.randint(10000),
                                      bthigh_joint_stiffness=bthigh_joint_stiffness,
                                      bshin_joint_stiffness=bshin_joint_stiffness,
                                      bfoot_joint_stiffness=bfoot_joint_stiffness)
                    done = False
                else:
                    state = env.reset(bthigh_joint_stiffness=bthigh_joint_stiffness,
                                      bshin_joint_stiffness=bshin_joint_stiffness,
                                      bfoot_joint_stiffness=bfoot_joint_stiffness)
                    done = False
                episode_reward = 0.0
                while not done:
                    if i == 0: # save video
                        images.append(img)
                        
                    action = predict(policy, state, env_action_type)
                    state, reward, done, _ = env.step(action)
                    episode_reward += reward
                    
                    if i == 0: # save video
                        img = env.render(mode='rgb_array')
                # episode done
                rewards.append(episode_reward)
                if i == 0: # save video
                    imageio.mimsave(curr_videopath, 
                                    [np.array(img) for i, 
                                    img in enumerate(images) if i%2==0],
                                    fps=29)      
            # episodes for current p are done
            avg_reward = np.sum(rewards) / eval_episodes
            print("---------------------------------------")
            print(f' back joints stiffness with p {p}')
            print(f" over {eval_episodes} episodes: {avg_reward:.3f}")
            print("---------------------------------------")
            avgs.append(avg_reward)
            stds.append(np.std(rewards))
        # all p's are done
        save_evals(save_path, setting, avgs, stds)
        
        # perturb 'front_joint_stiffness'
        setting = 'front_joint_stiffness'
        avgs = []
        stds = []
        for p in ps_fjntstif:
            rewards = []
            env.reset()
            fthigh_joint_stiffness = env.fthigh_joint_stiffness * (1 + p)
            fshin_joint_stiffness = env.fshin_joint_stiffness * (1 + p)
            ffoot_joint_stiffness = env.ffoot_joint_stiffness * (1 + p)
            for i in range(eval_episodes):
                if i == 0: # save video
                    images = []
                    img = env.render(mode='rgb_array')
                    curr_videopath = f'{video_path}/{setting}_{p:.2f}.gif'
                if hard:
                    state = env.reset(seed=np.random.randint(10000),
                                      fthigh_joint_stiffness=fthigh_joint_stiffness,
                                      fshin_joint_stiffness=fshin_joint_stiffness,
                                      ffoot_joint_stiffness=ffoot_joint_stiffness)
                    done = False
                else:
                    state = env.reset(fthigh_joint_stiffness=fthigh_joint_stiffness,
                                      fshin_joint_stiffness=fshin_joint_stiffness,
                                      ffoot_joint_stiffness=ffoot_joint_stiffness)
                    done = False
                episode_reward = 0.0
                while not done:
                    if i == 0: # save video
                        images.append(img)
                        
                    action = predict(policy, state, env_action_type)
                    state, reward, done, _ = env.step(action)
                    episode_reward += reward
                    
                    if i == 0: # save video
                        img = env.render(mode='rgb_array')
                # episode done
                rewards.append(episode_reward)
                if i == 0: # save video
                    imageio.mimsave(curr_videopath, 
                                    [np.array(img) for i, 
                                    img in enumerate(images) if i%2==0],
                                    fps=29)      
            # episodes for current p are done
            avg_reward = np.sum(rewards) / eval_episodes
            print("---------------------------------------")
            print(f' front joints stiffness with p {p}')
            print(f" over {eval_episodes} episodes: {avg_reward:.3f}")
            print("---------------------------------------")
            avgs.append(avg_reward)
            stds.append(np.std(rewards))
        # all p's are done
        save_evals(save_path, setting, avgs, stds)
        
        
        # perturb 'front_actuator_ctrlrange'
        setting = 'front_actuator_ctrlrange'
        avgs = []
        stds = []
        for x in xs_fact:
            rewards = []
            for i in range(eval_episodes):
                if i == 0: # save video
                    images = []
                    img = env.render(mode='rgb_array')
                    curr_videopath = f'{video_path}/{setting}_{x:.2f}.gif'
                if hard:
                    state, done = env.reset(seed=np.random.randint(10000),
                                            factuator_ctrlrange=(-x, x)), False
                else:
                    state, done = env.reset(factuator_ctrlrange=(-x, x)), False
                episode_reward = 0.0
                while not done:
                    if i == 0: # save video
                        images.append(img)
                    action = predict(policy, state, env_action_type)
                    state, reward, done, _ = env.step(action)
                    episode_reward += reward
                    
                    if i == 0: # save video
                        img = env.render(mode='rgb_array')
                # episode done
                rewards.append(episode_reward)
                if i == 0: # save video
                    imageio.mimsave(curr_videopath, 
                                    [np.array(img) for i, 
                                    img in enumerate(images) if i%2==0],
                                    fps=29)      
            # episodes for current p are done
            avg_reward = np.sum(rewards) / eval_episodes
            print("---------------------------------------")
            print(f'front_actuator_ctrlrange is [-{x}, {x}]')
            print(f" over {eval_episodes} episodes: {avg_reward:.3f}")
            print("---------------------------------------")
            avgs.append(avg_reward)
            stds.append(np.std(rewards))
        # all p's are done
        save_evals(save_path, setting, avgs, stds)
        
        # perturb 'back_actuator_ctrlrange'
        setting = 'back_actuator_ctrlrange'
        avgs = []
        stds = []
        for x in xs_bact:
            rewards = []
            for i in range(eval_episodes):
                if i == 0: # save video
                    images = []
                    img = env.render(mode='rgb_array')
                    curr_videopath = f'{video_path}/{setting}_{x:.2f}.gif'
                if hard:
                    state, done = env.reset(seed=np.random.randint(10000),
                                            factuator_ctrlrange=(-x, x)), False
                else:
                    state, done = env.reset(factuator_ctrlrange=(-x, x)), False
                episode_reward = 0.0
                while not done:
                    if i == 0: # save video
                        images.append(img)
                        
                    action = predict(policy, state, env_action_type)
                    state, reward, done, _ = env.step(action)
                    episode_reward += reward
                    
                    if i == 0: # save video
                        img = env.render(mode='rgb_array')
                # episode done
                rewards.append(episode_reward)
                if i == 0: # save video
                    imageio.mimsave(curr_videopath, 
                                    [np.array(img) for i, 
                                    img in enumerate(images) if i%2==0],
                                    fps=29)      
            # episodes for current p are done
            avg_reward = np.sum(rewards) / eval_episodes
            print("---------------------------------------")
            print(f' back_actuator_ctrlrange is [-{x}, {x}]')
            print(f" over {eval_episodes} episodes: {avg_reward:.3f}")
            print("---------------------------------------")
            avgs.append(avg_reward)
            stds.append(np.std(rewards))
        # all p's are done
        save_evals(save_path, setting, avgs, stds)
        
        # perturb 'joint_damping'
        setting = 'joint_damping'
        avgs = []
        stds = []
        for p in ps_damp:
            rewards = []
            for i in range(eval_episodes):
                if i == 0: # save video
                    images = []
                    img = env.render(mode='rgb_array')
                    curr_videopath = f'{video_path}/{setting}_{x:.2f}.gif'
                if hard:
                    state, done = env.reset(seed=np.random.randint(10000),
                                            joint_damping_p=p), False
                else:
                    state, done = env.reset(joint_damping_p=p), False
                episode_reward = 0.0
                while not done:
                    if i == 0: # save video
                        images.append(img)
                    action = predict(policy, state, env_action_type)
                    state, reward, done, _ = env.step(action)
                    episode_reward += reward
                    
                    if i == 0: # save video
                        img = env.render(mode='rgb_array')
                # episode done
                rewards.append(episode_reward)
                if i == 0: # save video
                    imageio.mimsave(curr_videopath, 
                                    [np.array(img) for i, 
                                    img in enumerate(images) if i%2==0],
                                    fps=29)      
            # episodes for current p are done
            avg_reward = np.sum(rewards) / eval_episodes
            print("---------------------------------------")
            print(f'joint damping with p {p}')
            print(f" over {eval_episodes} episodes: {avg_reward:.3f}")
            print("---------------------------------------")
            avgs.append(avg_reward)
            stds.append(np.std(rewards))
        # all p's are done
        save_evals(save_path, setting, avgs, stds)        
        
        
        # perturb 'joint_frictionloss'
        setting = 'joint_frictionloss'
        avgs = []
        stds = []
        for x in xs_fric:
            rewards = []
            for i in range(eval_episodes):
                if i == 0: # save video
                    images = []
                    img = env.render(mode='rgb_array')
                    curr_videopath = f'{video_path}/{setting}_{x:.2f}.gif'
                if hard:
                    state, done = env.reset(seed=np.random.randint(10000),
                                            joint_frictionloss=x), False
                else:
                    state, done = env.reset(joint_frictionloss=x), False
                episode_reward = 0.0
                while not done:
                    if i == 0: # save video
                        images.append(img)
                        
                    action = predict(policy, state, env_action_type)
                    state, reward, done, _ = env.step(action)
                    episode_reward += reward
                    
                    if i == 0: # save video
                        img = env.render(mode='rgb_array')
                # episode done
                rewards.append(episode_reward)
                if i == 0: # save video
                    imageio.mimsave(curr_videopath, 
                                    [np.array(img) for i, 
                                    img in enumerate(images) if i%2==0],
                                    fps=29)      
            # episodes for current p are done
            avg_reward = np.sum(rewards) / eval_episodes
            print("---------------------------------------")
            print(f'joint frictionloss with x {x}')
            print(f" over {eval_episodes} episodes: {avg_reward:.3f}")
            print("---------------------------------------")
            avgs.append(avg_reward)
            stds.append(np.std(rewards))
        # all p's are done
        save_evals(save_path, setting, avgs, stds) 
        
        # perturb actions
        setting = 'action'
        avgs = []
        stds = []
        for e in es:
            env.reset()
            rewards = []
            for i in range(eval_episodes):
                if i == 0: # save video
                    images = []
                    img = env.render(mode='rgb_array')
                    curr_videopath = f'{video_path}/{setting}_{e:.2f}.gif'
                state, done = env.reset(seed=np.random.randint(100000)), False
                episode_reward = 0.0
                while not done:
                    if i == 0: # save video
                        images.append(img)
                        
                    if np.random.binomial(n=1, p=e):
                        action = env.action_space.sample()
                    else: # else we use policy
                        action = predict(policy, state, env_action_type)
                    state, reward, done, _ = env.step(action)
                    episode_reward += reward
                    
                    if i == 0: # save video
                        img = env.render(mode='rgb_array')
                # episode done
                rewards.append(episode_reward)
                if i == 0: # save video
                    imageio.mimsave(curr_videopath, 
                                    [np.array(img) for i, 
                                    img in enumerate(images) if i%2==0],
                                    fps=29)
            # episodes for current p are done
            avg_reward = np.sum(rewards) / eval_episodes
            print("---------------------------------------")
            print(f' action with e {e}')
            print(f" over {eval_episodes} episodes: {avg_reward:.3f}")
            print("---------------------------------------")
            avgs.append(avg_reward)
            stds.append(np.std(rewards))
        # all p's are done
        save_evals(save_path, setting, avgs, stds)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # epsilon used in generating the offline data
    parser.add_argument('--data_eps', default=0.5, type=float)
    # environment name
    parser.add_argument('--env', default='CartPole-v0', type=str)
    parser.add_argument('--eval_episodes', default=20, type=int)
    parser.add_argument('--seed', default=1024, type=int)
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--rho', default=0.5, type=float)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--tau', default=0.005, type=float)
    parser.add_argument('--lmbda', default=0.75, type=float)                  
    parser.add_argument('--phi', default=0.1, type=float)
    parser.add_argument('--hard', default='True', type=str)
    # if model has extra commentary
    parser.add_argument('--comment', default='', type=str)
    # use d4rl dataset
    parser.add_argument("--d4rl", default='False', type=str)
    parser.add_argument("--d4rl_v2", default='False', type=str)
    parser.add_argument("--d4rl_expert", default='False', type=str)
    # use mixed policy
    parser.add_argument("--mixed", default='False', type=str)
    # policy used to generate data
    parser.add_argument("--gendata_pol", default='sac', type=str)

    args = parser.parse_args()

    if not os.path.exists("./perturbed_results"):
        os.makedirs("./perturbed_results")

    if not os.path.exists("./perturbed_videos"):
        os.makedirs("./perturbed_videos")
         
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
    
    # 2. determine state dimensions
    if isinstance(env.observation_space, spaces.Discrete):
        state_dim = 1
        max_state = env.observation_space.n - 1
    else:
        state_dim = env.observation_space.shape[0]
        max_state = np.inf
        
    # 3. check device
    if args.device not in ['cpu', 'cuda', 'cuda:0', 'cuda:1', 'auto']:
        raise NotImplementedError
        
    # check hard or not
    if args.hard == 'False':
        hard = False
    else:
        hard = True
        
    # get perturbed environment
    i = args.env.find('-')
    perturbed_env = f'{args.env[:i]}Perturbed{args.env[i:]}'
        
    # check d4rl option and mixed option
    # determine data_path, log_path and save_path
    if args.d4rl == 'False' and args.mixed == 'False':
        save_path = f'RFQI_{perturbed_env}_rho{args.rho}_dataeps{args.data_eps}_datapol{args.gendata_pol}{args.comment}'
        load_path = f'RFQI_{args.env}_rho{args.rho}_dataeps{args.data_eps}_datapol{args.gendata_pol}{args.comment}'
    elif args.d4rl == 'True' and args.mixed == 'False':
        save_path = f'RFQI_{perturbed_env}_rho{args.rho}_d4rl'
        load_path = f'RFQI_{args.env}_rho{args.rho}_d4rl'
        if args.d4rl_expert == 'True':
            save_path += '_expert'
            load_path += '_expert'
        save_path += args.comment
        load_path += args.comment
    elif args.d4rl == 'False' and args.mixed == 'True':
        save_path = f'RFQI_mixed_{perturbed_env}_rho{args.rho}_dataeps{args.data_eps}_datapol{args.gendata_pol}{args.comment}'
        load_path = f'RFQI_mixed_{args.env}_rho{args.rho}_dataeps{args.data_eps}_datapol{args.gendata_pol}{args.comment}'
    else:
        raise NotImplementedError
    paths = dict(load_path=load_path,
                 save_path=save_path)    

    print("=========================================================")
    print(f'===============Eval. RFQI on {perturbed_env}==============')
    print(f'{args.env} attributes: max_action={max_action}')
    print(f'                       min_action={min_action}')
    print(f'                       action_dim={action_dim}')
    print(f'                       env action type is {env_action_type}')
    print(f'                       extra comment: {args.comment}')
    print(f'Eval. attributes:      using device: {args.device}')
    print(f'                       eval episodes: {args.eval_episodes}')
    print(f'RFQI attributes:       rho={args.rho}')
    if args.d4rl == 'True':
        print(f'                       trained on d4rl')
    elif args.mixed == 'True':
        print(f'                       trained on data with eps={args.data_eps}')
        print(f'                       data collecting policy is WITH mixed')
    else:
        print(f'                       trained on data with eps={args.data_eps}')
    print(f'                       loading from: {load_path}')    
    print("=========================================================")
    eval_rfqi(state_dim, action_dim, min_action, max_action, hard, paths,
              perturbed_env, args)
        
