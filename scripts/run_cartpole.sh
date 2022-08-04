python train_rfqi.py --data_eps=0.3 --env=CartPole-v0 --max_trn_steps=2000 --eval_freq=50 --device=cuda --data_size=1000000 --batch_size=100 --rho=0.5 --gendata_pol=ppo

python eval_rfqi.py --data_eps=0.3 --gendata_pol=ppo --env=CartPole-v0 --rho=0.5
