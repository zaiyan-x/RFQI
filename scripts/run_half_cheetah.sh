python train_rfqi.py --env=HalfCheetah-v3 --max_trn_steps=500000 --eval_freq=1000 --device=cuda --data_size=1000000 --batch_size=1000 --rho=0.3 --actor_lr=3e-4 --critic_lr=8e-4 --adam_eps=1e-6 --adam_lr=1e-3 --eval_episodes=10 --d4rl=True --d4rl_v2=True

python eval_rfqi.py --env=HalfCheetah-v3 --device=cuda --rho=0.3 --d4rl=True --d4rl_v2=True
