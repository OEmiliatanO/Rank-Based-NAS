# NASSPACE=nasbench201 # [nasbench101, nasbench201, natbenchsss]
# SEARCH_ALGO=RD # [RD, GA, SA]
# API_LOC=./NAS-Bench-201.pth # [./NAS-Bench-201.pth, ../NATS-sss-v1_0-50262-simple/]
# DATASET=cifar100 # [cifar10, cifar100, ImageNet16-120]
# DATA_LOC=./cifardata/ # [./cifardata/, ../ImageNet16/]

# search_algo SA_rk random
# SA_rk spec: end_T maxn_iter Rt init_T

# ==================================================================================

# natsbenchsss
# RD
# cifar10
#python RD_search.py --save_string RD --augtype none --repeat 1 --sigma 1 --nasspace natsbenchsss --api_loc ../NATS-sss-v1_0-50262-simple/ --batch_size 128 --GPU 0 --n_samples 1000 --n_runs 100 --data_loc ./cifardata/ --dataset cifar10 --valid
#python RD_search.py --save_string RD --augtype none --repeat 1 --sigma 1 --nasspace natsbenchsss --api_loc ../NATS-sss-v1_0-50262-simple/ --batch_size 128 --GPU 0 --n_samples 1000 --n_runs 100 --data_loc ./cifardata/ --dataset cifar10 --test
# cifar100
#python RD_search.py --save_string RD --augtype none --repeat 1 --sigma 1 --nasspace natsbenchsss --api_loc ../NATS-sss-v1_0-50262-simple/ --batch_size 128 --GPU 0 --n_samples 1000 --n_runs 100 --data_loc ./cifardata/ --dataset cifar100 --valid
#python RD_search.py --save_string RD --augtype none --repeat 1 --sigma 1 --nasspace natsbenchsss --api_loc ../NATS-sss-v1_0-50262-simple/ --batch_size 128 --GPU 0 --n_samples 1000 --n_runs 100 --data_loc ./cifardata/ --dataset cifar100 --test
# ImageNet16-120
#python RD_search.py --save_string RD --augtype none --repeat 1 --sigma 1 --nasspace natsbenchsss --api_loc ../NATS-sss-v1_0-50262-simple/ --batch_size 128 --GPU 0 --n_samples 1000 --n_runs 100 --data_loc ../ImageNet16/ --dataset ImageNet16-120 --valid
#python RD_search.py --save_string RD --augtype none --repeat 1 --sigma 1 --nasspace natsbenchsss --api_loc ../NATS-sss-v1_0-50262-simple/ --batch_size 128 --GPU 0 --n_samples 1000 --n_runs 100 --data_loc ../ImageNet16/ --dataset ImageNet16-120 --test

# SA
# cifar10
#python SA_search.py --end_T 8e-4 --maxn_iter 4 --Rt 0.745 --init_T 1 --maxN 10 --alpha 0.25 --save_string SA --augtype none --repeat 1 --sigma 1 --nasspace natsbenchsss --api_loc ../NATS-sss-v1_0-50262-simple/ --batch_size 128 --GPU 0 --n_samples 50 --n_runs 100 --data_loc ./cifardata/ --dataset cifar10 --valid
#python SA_search.py --end_T 8e-4 --maxn_iter 4 --Rt 0.745 --init_T 1 --maxN 10 --alpha 0.25 --save_string SA --augtype none --repeat 1 --sigma 1 --nasspace natsbenchsss --api_loc ../NATS-sss-v1_0-50262-simple/ --batch_size 128 --GPU 0 --n_samples 50 --n_runs 100 --data_loc ./cifardata/ --dataset cifar10 --test
# cifar100
#python SA_search.py --end_T 8e-4 --maxn_iter 4 --Rt 0.745 --init_T 1 --maxN 10 --alpha 0.25 --save_string SA --augtype none --repeat 1 --sigma 1 --nasspace natsbenchsss --api_loc ../NATS-sss-v1_0-50262-simple/ --batch_size 128 --GPU 0 --n_samples 50 --n_runs 100 --data_loc ./cifardata/ --dataset cifar100 --valid
#python SA_search.py --end_T 8e-4 --maxn_iter 4 --Rt 0.745 --init_T 1 --maxN 10 --alpha 0.25 --save_string SA --augtype none --repeat 1 --sigma 1 --nasspace natsbenchsss --api_loc ../NATS-sss-v1_0-50262-simple/ --batch_size 128 --GPU 0 --n_samples 50 --n_runs 100 --data_loc ./cifardata/ --dataset cifar100 --test
# ImageNet16-120
#python SA_search.py --end_T 8e-4 --maxn_iter 4 --Rt 0.745 --init_T 1 --maxN 10 --alpha 0.25 --save_string SA --augtype none --repeat 1 --sigma 1 --nasspace natsbenchsss --api_loc ../NATS-sss-v1_0-50262-simple/ --batch_size 128 --GPU 0 --n_samples 50 --n_runs 100 --data_loc ../ImageNet16/ --dataset ImageNet16-120 --valid
#python SA_search.py --end_T 8e-4 --maxn_iter 4 --Rt 0.745 --init_T 1 --maxN 10 --alpha 0.25 --save_string SA --augtype none --repeat 1 --sigma 1 --nasspace natsbenchsss --api_loc ../NATS-sss-v1_0-50262-simple/ --batch_size 128 --GPU 0 --n_samples 50 --n_runs 100 --data_loc ../ImageNet16/ --dataset ImageNet16-120 --test

# nasbench201
# RD
# cifar10
#python RD_search.py --save_string RD --augtype none --repeat 1 --sigma 1 --nasspace nasbench201 --api_loc ../NATS-tss-v1_0-3ffb9-simple/ --batch_size 128 --GPU 0 --n_samples 1000 --n_runs 100 --data_loc ./cifardata/ --dataset cifar10 --valid
#python RD_search.py --save_string RD --augtype none --repeat 1 --sigma 1 --nasspace nasbench201 --api_loc ../NATS-tss-v1_0-3ffb9-simple/ --batch_size 128 --GPU 0 --n_samples 1000 --n_runs 100 --data_loc ./cifardata/ --dataset cifar10 --test
# cifar100
#python RD_search.py --save_string RD --augtype none --repeat 1 --sigma 1 --nasspace nasbench201 --api_loc ../NATS-tss-v1_0-3ffb9-simple/ --batch_size 128 --GPU 0 --n_samples 1000 --n_runs 100 --data_loc ./cifardata/ --dataset cifar100 --valid
#python RD_search.py --save_string RD --augtype none --repeat 1 --sigma 1 --nasspace nasbench201 --api_loc ../NATS-tss-v1_0-3ffb9-simple/ --batch_size 128 --GPU 0 --n_samples 1000 --n_runs 100 --data_loc ./cifardata/ --dataset cifar100 --test
# ImageNet16-120
#python RD_search.py --save_string RD --augtype none --repeat 1 --sigma 1 --nasspace nasbench201 --api_loc ../NATS-tss-v1_0-3ffb9-simple/ --batch_size 128 --GPU 0 --n_samples 1000 --n_runs 100 --data_loc ../ImageNet16/ --dataset ImageNet16-120 --valid
#python RD_search.py --save_string RD --augtype none --repeat 1 --sigma 1 --nasspace nasbench201 --api_loc ../NATS-tss-v1_0-3ffb9-simple/ --batch_size 128 --GPU 0 --n_samples 1000 --n_runs 100 --data_loc ../ImageNet16/ --dataset ImageNet16-120 --test

# SA
# cifar10
#python SA_search.py --end_T 8e-4 --maxn_iter 4 --Rt 0.745 --init_T 1 --maxN 10 --alpha 0.25 --save_string SA --augtype none --repeat 1 --sigma 1 --nasspace nasbench201  --api_loc ../NATS-tss-v1_0-3ffb9-simple/ --batch_size 128 --GPU 0 --n_samples 50 --n_runs 100 --data_loc ./cifardata/ --dataset cifar10 --valid
#python SA_search.py --end_T 8e-4 --maxn_iter 4 --Rt 0.745 --init_T 1 --maxN 10 --alpha 0.25 --save_string SA --augtype none --repeat 1 --sigma 1 --nasspace nasbench201  --api_loc ../NATS-tss-v1_0-3ffb9-simple/ --batch_size 128 --GPU 0 --n_samples 50 --n_runs 100 --data_loc ./cifardata/ --dataset cifar10 --test
# cifar100
#python SA_search.py --end_T 8e-4 --maxn_iter 4 --Rt 0.745 --init_T 1 --maxN 10 --alpha 0.25 --save_string SA --augtype none --repeat 1 --sigma 1 --nasspace nasbench201  --api_loc ../NATS-tss-v1_0-3ffb9-simple/ --batch_size 128 --GPU 0 --n_samples 50 --n_runs 100 --data_loc ./cifardata/ --dataset cifar100 --valid
#python SA_search.py --end_T 8e-4 --maxn_iter 4 --Rt 0.745 --init_T 1 --maxN 10 --alpha 0.25 --save_string SA --augtype none --repeat 1 --sigma 1 --nasspace nasbench201  --api_loc ../NATS-tss-v1_0-3ffb9-simple/ --batch_size 128 --GPU 0 --n_samples 50 --n_runs 100 --data_loc ./cifardata/ --dataset cifar100 --test
# ImageNet16-120
#python SA_search.py --end_T 8e-4 --maxn_iter 4 --Rt 0.745 --init_T 1 --maxN 10 --alpha 0.25 --save_string SA --augtype none --repeat 1 --sigma 1 --nasspace nasbench201  --api_loc ../NATS-tss-v1_0-3ffb9-simple/ --batch_size 128 --GPU 0 --n_samples 50 --n_runs 100 --data_loc ../ImageNet16/ --dataset ImageNet16-120 --valid
#python SA_search.py --end_T 8e-4 --maxn_iter 4 --Rt 0.745 --init_T 1 --maxN 10 --alpha 0.25 --save_string SA --augtype none --repeat 1 --sigma 1 --nasspace nasbench201  --api_loc ../NATS-tss-v1_0-3ffb9-simple/ --batch_size 128 --GPU 0 --n_samples 50 --n_runs 100 --data_loc ../ImageNet16/ --dataset ImageNet16-120 --test

# nasbench101
# RD
# cifar10
#python RD_search.py --save_string RD --num_labels 10 --augtype none --repeat 1 --sigma 1 --nasspace nasbench101 --api_loc ./nasbench_full.tfrecord --batch_size 128 --GPU 0 --n_samples 1000 --n_runs 100 --data_loc ./cifardata/ --dataset cifar10 --valid
#python RD_search.py --save_string RD --num_labels 10 --augtype none --repeat 1 --sigma 1 --nasspace nasbench101 --api_loc ./nasbench_full.tfrecord --batch_size 128 --GPU 0 --n_samples 1000 --n_runs 100 --data_loc ./cifardata/ --dataset cifar10 --test

# SA
# cifar10
#python SA_search.py --end_T 8e-4 --maxn_iter 4 --Rt 0.745 --init_T 1 --maxN 10 --alpha 0.25 --save_string SA --num_labels 10 --augtype none --repeat 1 --sigma 1 --nasspace nasbench101 --api_loc ./nasbench_full.tfrecord --batch_size 128 --GPU 0 --n_samples 50 --n_runs 100 --data_loc ./cifardata/ --dataset cifar10 --valid
#python SA_search.py --end_T 8e-4 --maxn_iter 4 --Rt 0.745 --init_T 1 --maxN 10 --alpha 0.25 --save_string SA --num_labels 10 --augtype none --repeat 1 --sigma 1 --nasspace nasbench101 --api_loc ./nasbench_full.tfrecord --batch_size 128 --GPU 0 --n_samples 50 --n_runs 100 --data_loc ./cifardata/ --dataset cifar10 --test
