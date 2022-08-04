import argparse
import gym
from gym import spaces
import numpy as np
import os
import torch
import imageio

from fqi import PQL_BCQ


def predict(policy, state, env_action_type):
    '''
    PQL version of predict.
    '''
    action = policy.select_action(np.array(state))
    if env_action_type == 'discrete':
        return np.rint(action[0]).astype(int)
    elif env_action_type == 'continuous':
        return action[0]
    elif env_action_type == 'multi_continuous':
        return action
    else:
        raise NotImplementedError


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

def load_policy(policy, load_path, device):
    policy.actor.load(f'{load_path}_actor', device=device)
    policy.critic.load(f'{load_path}_critic', device=device)
    policy.vae.load(f'{load_path}_action_vae', device=device)
    policy.vae2.load(f'{load_path}_state_vae', device=device)   
    return policy

def save_evals(save_path, setting, avgs, stds):
    np.save(f'{save_path}_{setting}_avgs', avgs)
    np.save(f'{save_path}_{setting}_stds', stds)


def eval_FQI(state_dim, action_dim, max_state, min_action, max_action,
             hard, perturbed_env, args, eval_episodes=20):
    '''
    Evaluate PQL on perturbed environments.
    '''
    # parse paths
    load_path = f"./models/{paths['load_path']}"
    save_path = f"./perturbed_results/{paths['save_path']}"
    video_path = f"./perturbed_videos/{paths['save_path']}"
    if not os.path.exists(video_path):
        os.makedirs(video_path)
    
    # Initialize policy
    policy = PQL_BCQ(state_dim, action_dim, max_state, min_action, max_action,
                     args.device, args.discount, args.tau, args.lmbda, args.phi,
                     n_action=args.n_action,
                     n_action_execute=args.n_action_execute,
                     backup=args.backup, ql_noise=args.ql_noise,
                     actor_lr=args.actor_lr, beta=args.beta, vmin=args.vmin)
    policy = load_policy(policy, load_path, args.device)
    
    # make environment and identify settings to be perturbed
    env = gym.make(perturbed_env)
    # get the action type
    env_action_type = get_action_type(env.action_space)
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
                state, done = env.reset(seed=np.random.randint(100000),
                                        force_mag=force_mag, 
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
                state, done = env.reset(seed=np.random.randint(100000),
                                        gravity=gravity, 
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
                state, done = env.reset(seed=np.random.randint(100000),
                                        length=length, 
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
                state, done = env.reset(seed=np.random.randint(100000)), False
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
    # OpenAI gym environment name (need to be consistent with the dataset name)
    parser.add_argument("--env", default="Hopper-v2")
    # sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--seed", default=1, type=int)  
    # how often (in time steps) we evaluate
    parser.add_argument("--eval_episodes", default=20, type=int)
    # evaluate on randomly initialized environment every episode
    parser.add_argument("--hard", default='True', type=str)
    # the epsilon used to generate the training data
    parser.add_argument("--data_eps", default=0.3, type=float)
    # use d4rl dataset
    parser.add_argument("--d4rl", default='False', type=str)
    parser.add_argument("--d4rl_v2", default='False', type=str)
    parser.add_argument("--d4rl_expert", default='False', type=str)
    # use mixed policy
    parser.add_argument("--mixed", default='False', type=str)
    # policy used to generate data
    parser.add_argument("--gendata_pol", default='sac', type=str)
    # check policy comment
    parser.add_argument("--comment", default='', type=str)
    # device to run evaluations
    parser.add_argument("--device", default='cpu', type=str)
    
    #==========================BCQ parameter==========================
    # discount factor
    parser.add_argument("--discount", default=0.99)  
    # Target network update rate
    parser.add_argument("--tau", default=0.005)  
    # Weighting for clipped double Q-learning in BCQ
    parser.add_argument("--lmbda", default=0.75)  
    # Max perturbation hyper-parameter for BCQ
    parser.add_argument("--phi", default=0.1, type=float)
    # learning rate of actor
    parser.add_argument("--actor_lr", default=1e-3, type=float) 
    # number of sampling action for policy (in backup)
    parser.add_argument("--n_action", default=100, type=int) 
    # number of sampling action for policy (in execution)
    parser.add_argument("--n_action_execute", default=100, type=int) 

    #==========================PQL-BCQ parameter=======================
    parser.add_argument("--backup", type=str, default="QL") # "QL": q learning (Q-max) back up, "AC": actor-critic backup
    parser.add_argument("--ql_noise", type=float, default=0.15) # Noise of next action in QL
    parser.add_argument("--automatic_beta", type=str, default='True')  # If true, use percentile for b (beta is the b in paper)
    parser.add_argument("--beta_percentile", type=float, default=2.0)  # Use x-Percentile as the value of b
    parser.add_argument("--beta", default=-0.4, type=float)  # hardcoded b, only effective when automatic_beta = False
    parser.add_argument("--vmin", default=0, type=float) # min value of the environment. Empirically I set it to be the min of 1000 random rollout.


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
    
    # 4. check hard or not
    if args.hard == 'False':
        hard = False
    else:
        hard = True
        
    # 5. check d4rl option and mixed option
    # determine data_path, log_path and save_path
    # get perturbed environment
    i = args.env.find('-')
    perturbed_env = f'{args.env[:i]}Perturbed{args.env[i:]}'  
    if args.d4rl == 'False' and args.mixed == 'False':
        save_path = f'FQI_{perturbed_env}_dataeps{args.data_eps}_datapol{args.gendata_pol}{args.comment}'
        load_path = f'FQI_{args.env}_dataeps{args.data_eps}_datapol{args.gendata_pol}{args.comment}'
    elif args.d4rl == 'True' and args.mixed == 'False':  
        save_path = f'FQI_{perturbed_env}_d4rl'
        load_path = f'FQI_{args.env}_d4rl'
        if args.d4rl_expert == 'True':
            save_path += '_expert'
            load_path += '_expert'
        save_path += args.comment
        load_path += args.comment
    elif args.d4rl == 'False' and args.mixed == 'True':
        save_path = f'FQI_mixed_{perturbed_env}_dataeps{args.data_eps}_datapol{args.gendata_pol}{args.comment}'
        load_path = f'FQI_mixed_{args.env}_dataeps{args.data_eps}_datapol{args.gendata_pol}{args.comment}'
    else:
        raise NotImplementedError
    paths = dict(load_path=load_path,
                 save_path=save_path) 
        
    # broadcast
    i = args.env.find('-')
    perturbed_env = f'{args.env[:i]}Perturbed{args.env[i:]}'
    print("=========================================================")
    print(f'===============Eval. FQI on {perturbed_env}==============')
    print(f'{args.env} attributes: max_action={max_action}')
    print(f'                       min_action={min_action}')
    print(f'                       action_dim={action_dim}')
    print(f'                       env action type is {env_action_type}')
    print(f'Eval. attributes:      using device: {args.device}')
    print(f'                       eval episodes: {args.eval_episodes}')
    print(f'FQI attributes:        beta_percentile={args.beta_percentile}')
    if args.d4rl == 'True':
        print(f'                       trained on d4rl')
    elif args.mixed == 'True':
        print(f'                       trained on data with eps={args.data_eps}')
        print(f'                       data collecting policy is WITH mixed')
    print(f'                       extra comment: {args.comment}')
    print("=========================================================")
    eval_FQI(state_dim, action_dim, max_state, min_action, max_action, hard,
             perturbed_env, args)
 
        
