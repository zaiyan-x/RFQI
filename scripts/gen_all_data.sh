python generate_offline_data.py --env=CartPole-v0 --gendata_pol=ppo --eps=0.3

python generate_offline_data.py --env=Hopper-v3 --gendata_pol=sac --eps=0.3 --mixed=True

python load_d4rl_data.py
