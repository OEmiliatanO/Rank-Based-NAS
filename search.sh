#python GA_search.py --search_algo rk --maxn_pop 100 --maxn_iter 40 --prob_mut 0.08 --prob_cr 0.7 --save_string all_score --augtype none --repeat 1 --sigma 0.05 --nasspace nasbench201 --batch_size 128 --GPU 0 --n_samples 50 --n_runs 50 --data_loc ./cifardata/ --dataset cifar100 --test
#python GA_search.py --search_algo rk --maxn_pop 20 --maxn_iter 60 --prob_mut 0.08 --prob_cr 0.7 --save_string all_score --test --augtype none --repeat 1 --sigma 0.05 --nasspace nasbench201 --batch_size 128 --GPU 0 --n_samples 50 --data_loc ../ImageNet16 --dataset ImageNet16-120

python search.py --save_string all_score --augtype none --repeat 1 --sigma 1 --nasspace nasbench201 --batch_size 128 --GPU 0 --data_loc ./cifardata/ --dataset cifar100 --test --n_samples 1000 --n_runs 50
#python search.py --save_string all_score --augtype none --repeat 1 --sigma 1 --nasspace nasbench201 --batch_size 128 --GPU 0 --data_loc ../ImageNet16 --dataset ImageNet16-120 --test --n_samples 1000 --n_runs 50
