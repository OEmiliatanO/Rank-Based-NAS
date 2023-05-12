# NASSPACE=nasbench201 # [nasbench101, nasbench201, natbenchsss]
# SEARCH_ALGO=RD # [RD, GA, SA]
# API_LOC=./NAS-Bench-201.pth # [./NAS-Bench-201.pth, ../NATS-sss-v1_0-50262-simple/]
# DATASET=cifar100 # [cifar10, cifar100, ImageNet16-120]
# DATA_LOC=./cifardata/ # [./cifardata/, ../ImageNet16/]

# search_algo GA_rk SA_rk random
# GA_rk spec: maxn_pop maxn_iter prob_mut prob_cr 
# SA_rk spec: end_T maxn_iter Rt init_T
#python RD_search.py --save_string RD --sigma 1 --nasspace natsbenchsss --api_loc ../NATS-sss-v1_0-50262-simple/ --batch_size 128 --GPU 0 --n_samples 1000 --n_runs 50 --data_loc ./cifardata/ --dataset cifar100 --test
#python RD_search.py --save_string RD --sigma 1 --nasspace natsbenchsss --api_loc ../NATS-sss-v1_0-50262-simple/ --batch_size 128 --GPU 0 --n_samples 1000 --n_runs 50 --data_loc ../ImageNet16/ --dataset ImageNet16-120 --test

# natsbenchsss
# GA
# cifar10
#python GA_search.py --maxn_pop 100 --maxn_iter 10 --prob_mut 0.08 --prob_cr 0.7 --save_string GA --augtype none --repeat 1 --sigma 1 --nasspace natsbenchsss --api_loc ../NATS-sss-v1_0-50262-simple/ --batch_size 128 --GPU 0 --n_samples 50 --n_runs 50 --data_loc ./cifardata/ --dataset cifar10 --test
# cifar100
python GA_search.py --maxn_pop 100 --maxn_iter 10 --prob_mut 0.08 --prob_cr 0.7 --save_string GA --augtype none --repeat 1 --sigma 1 --nasspace natsbenchsss --api_loc ../NATS-sss-v1_0-50262-simple/ --batch_size 128 --GPU 0 --n_samples 50 --n_runs 50 --data_loc ./cifardata/ --dataset cifar100 --test
# ImageNet16-120
#python GA_search.py --maxn_pop 100 --maxn_iter 10 --prob_mut 0.08 --prob_cr 0.7 --save_string GA --augtype none --repeat 1 --sigma 1 --nasspace natsbenchsss --api_loc ../NATS-sss-v1_0-50262-simple/ --batch_size 128 --GPU 0 --n_samples 50 --n_runs 50 --data_loc ../ImageNet16/ --dataset ImageNet16-120 --test

# RD
# cifar10
#python RD_search.py --save_string RD --augtype none --repeat 1 --sigma 1 --nasspace natsbenchsss --api_loc ../NATS-sss-v1_0-50262-simple/ --batch_size 128 --GPU 0 --n_samples 1000 --n_runs 50 --data_loc ./cifardata/ --dataset cifar10 --test
# cifar100
#python RD_search.py --save_string RD --augtype none --repeat 1 --sigma 1 --nasspace natsbenchsss --api_loc ../NATS-sss-v1_0-50262-simple/ --batch_size 128 --GPU 0 --n_samples 1000 --n_runs 50 --data_loc ./cifardata/ --dataset cifar100 --test
# ImageNet16-120
#python RD_search.py --save_string RD --augtype none --repeat 1 --sigma 1 --nasspace natsbenchsss --api_loc ../NATS-sss-v1_0-50262-simple/ --batch_size 128 --GPU 0 --n_samples 1000 --n_runs 50 --data_loc ../ImageNet16/ --dataset ImageNet16-120 --test

# SA
# cifar10
#python SA_search.py --end_T 1e-3 --maxn_iter 4 --Rt 0.6 --init_T 1 --maxN 10 --save_string SA --augtype none --repeat 1 --sigma 1 --nasspace natsbenchsss --api_loc ../NATS-sss-v1_0-50262-simple/ --batch_size 128 --GPU 0 --n_samples 50 --n_runs 50 --data_loc ./cifardata/ --dataset cifar10 --test
# cifar100
#python SA_search.py --end_T 1e-3 --maxn_iter 4 --Rt 0.6 --init_T 1 --maxN 10 --save_string SA --augtype none --repeat 1 --sigma 1 --nasspace natsbenchsss --api_loc ../NATS-sss-v1_0-50262-simple/ --batch_size 128 --GPU 0 --n_samples 50 --n_runs 50 --data_loc ./cifardata/ --dataset cifar100 --test
# ImageNet16-120
#python SA_search.py --end_T 1e-3 --maxn_iter 4 --Rt 0.6 --init_T 1 --maxN 10 --save_string SA --augtype none --repeat 1 --sigma 1 --nasspace natsbenchsss --api_loc ../NATS-sss-v1_0-50262-simple/ --batch_size 128 --GPU 0 --n_samples 50 --n_runs 50 --data_loc ../ImageNet16/ --dataset ImageNet16-120 --test

# nasbench201
# GA
# cifar10
#python GA_search.py --maxn_pop 100 --maxn_iter 40 --prob_mut 0.08 --prob_cr 0.7 --save_string GA --augtype none --repeat 1 --sigma 1 --nasspace nasbench201 --batch_size 128 --GPU 0 --n_samples 50 --n_runs 50 --data_loc ./cifardata --dataset cifar10 --test
# cifar100
#python GA_search.py --maxn_pop 100 --maxn_iter 40 --prob_mut 0.08 --prob_cr 0.7 --save_string GA --augtype none --repeat 1 --sigma 1 --nasspace nasbench201 --batch_size 128 --GPU 0 --n_samples 50 --n_runs 50 --data_loc ./cifardata --dataset cifar100 --test
# ImageNet16-120
#python GA_search.py --maxn_pop 100 --maxn_iter 40 --prob_mut 0.08 --prob_cr 0.7 --save_string GA --augtype none --repeat 1 --sigma 1 --nasspace nasbench201 --batch_size 128 --GPU 0 --n_samples 50 --n_runs 50 --data_loc ../ImageNet16/ --dataset ImageNet16-120 --test

# RD
# cifar10
#python RD_search.py --save_string RD --augtype none --repeat 1 --sigma 1 --nasspace nasbench201 --batch_size 128 --GPU 0 --n_samples 1000 --n_runs 50 --data_loc ./cifardata/ --dataset cifar10 --test
# cifar100
#python RD_search.py --save_string RD --augtype none --repeat 1 --sigma 1 --nasspace nasbench201 --batch_size 128 --GPU 0 --n_samples 1000 --n_runs 50 --data_loc ./cifardata/ --dataset cifar100 --test
# ImageNet16-120
#python RD_search.py --save_string RD --augtype none --repeat 1 --sigma 1 --nasspace nasbench201 --batch_size 128 --GPU 0 --n_samples 1000 --n_runs 50 --data_loc ../ImageNet16/ --dataset ImageNet16-120 --test

# SA
# cifar10
#python SA_search.py --end_T 1e-3 --maxn_iter 4 --Rt 0.6 --init_T 1 --maxN 10 --save_string SA --augtype none --repeat 1 --sigma 1 --nasspace nasbench201   --batch_size 128 --GPU 0 --n_samples 50 --n_runs 50 --data_loc ./cifardata/ --dataset cifar10 --test
# cifar100
#python SA_search.py --end_T 1e-3 --maxn_iter 4 --Rt 0.6 --init_T 1 --maxN 10 --save_string SA --augtype none --repeat 1 --sigma 1 --nasspace nasbench201   --batch_size 128 --GPU 0 --n_samples 50 --n_runs 50 --data_loc ./cifardata/ --dataset cifar100 --test
# ImageNet16-120
python SA_search.py --end_T 1e-3 --maxn_iter 4 --Rt 0.6 --init_T 1 --maxN 10 --save_string SA --augtype none --repeat 1 --sigma 1 --nasspace nasbench201   --batch_size 128 --GPU 0 --n_samples 50 --n_runs 50 --data_loc ../ImageNet16/ --dataset ImageNet16-120 --test
