python eval_rfqi.py --data_eps=0.3 --env=CartPole-v0 --device=cpu --rho=0.5 --gendata_pol=ppo

python eval_rfqi.py --env=Hopper-v3 --device=cpu --rho=0.5 --comment='_adamlr1e-3' --d4rl=True

python eval_rfqi.py --env=Hopper-v3 --device=cpu --rho=0.5 --data_eps=0.3 --mixed=True --gendata_pol=sac

python eval_rfqi.py --env=HalfCheetah-v3 --device=cpu --rho=0.5 --d4rl=True --comment='_adamlr1e-3'