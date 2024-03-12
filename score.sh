#python score.py --save_string score --augtype none --repeat 1 --sigma 1 --nasspace nasbench201 --api_loc ../NATS-tss-v1_0-3ffb9-simple/ --batch_size 128 --GPU 1 --n_runs 1 --n_samples 10 --data_loc ./cifardata/ --dataset cifar10 --test
#python score.py --save_string score --augtype none --repeat 1 --sigma 1 --nasspace nasbench201 --api_loc ../NATS-tss-v1_0-3ffb9-simple/ --batch_size 128 --GPU 1 --n_runs 1 --n_samples 100 --data_loc ./cifardata/ --dataset cifar10 --test
python score.py --save_string score --augtype none --repeat 1 --sigma 1 --nasspace nasbench201 --api_loc ../NATS-tss-v1_0-3ffb9-simple/ --batch_size 128 --GPU 1 --n_runs 1 --n_samples 1000 --data_loc ./cifardata/ --dataset cifar10 --test
python score_PCA.py results/score/score_nasbench201_cifar10_False_False_none_1.0_1_128_1_1000_False_True_1.t7
mv score_group1.png 201_train.png
mv score_group2.png 201_test.png

#python score.py --save_string score --augtype none --repeat 1 --sigma 1 --nasspace nasbench201 --api_loc ../NATS-tss-v1_0-3ffb9-simple/ --batch_size 128 --GPU 1 --n_runs 1 --n_samples 10000 --data_loc ./cifardata/ --dataset cifar10 --test

#python score.py --save_string score --augtype none --repeat 1 --sigma 1 --nasspace nasbench201 --api_loc ../NATS-tss-v1_0-3ffb9-simple/ --batch_size 128 --GPU 1 --n_runs 1 --n_samples 1000 --data_loc ../ImageNet16/ --dataset ImageNet16-120 --test

python score.py --save_string score --augtype none --repeat 1 --sigma 1 --nasspace natsbenchsss --api_loc ../NATS-sss-v1_0-50262-simple/ --batch_size 128 --GPU 1 --n_runs 1 --n_samples 1000 --data_loc ./cifardata/ --dataset cifar10 --test
python score_PCA.py results/score/score_natsbenchsss_cifar10_False_False_none_1.0_1_128_1_1000_False_True_1.t7
mv score_group1.png sss_train.png
mv score_group2.png sss_test.png

python score.py --save_string score --num_labels 10 --augtype none --repeat 1 --sigma 1 --nasspace nasbench101 --api_loc ./nasbench_full.tfrecord --batch_size 128 --GPU 1 --n_runs 1 --n_samples 1000 --data_loc ./cifardata/ --dataset cifar10 --test
python score_PCA.py results/score/score_nasbench101_cifar10_False_False_none_1.0_1_128_1_1000_False_True_1.t7
mv score_group1.png 101_train.png
mv score_group2.png 101_test.png
