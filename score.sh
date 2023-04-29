#python score_networks.py --save_loc results/score/cifar10 --save_string all_score --valid --augtype none --repeat 1 --sigma 0.05 --nasspace nasbench201 --batch_size 128 --GPU 0 --n_samples 50 --dataset cifar10

<<plot_example
python process_data.py --save_loc $1 --oper ninaswot_add_ntk_add_entropy --save_string all_score --valid --augtype none --repeat 1 --sigma 0.05 --nasspace nasbench201 --batch_size 128 --GPU 0 --n_samples 50 --dataset cifar10
python plot_2d_scores.py --save_loc $1 --targets acc-ninaswot_add_ntk_add_entropy --save_string all_score --valid --augtype none --repeat 1 --sigma 0.05 --nasspace nasbench201 --batch_size 128 --GPU 0 --n_samples 50 --dataset cifar10
python plot_3d_scores.py --save_loc $1 --save_string all_score --valid --augtype none --repeat 1 --sigma 0.05 --nasspace nasbench201 --batch_size 128 --GPU 0 --n_samples 50 --dataset cifar10
plot_example

#python score_networks.py --save_loc results/score/cifar100 --save_string all_score --valid --augtype none --repeat 1 --sigma 0.05 --nasspace nasbench201 --batch_size 128 --GPU 0 --dataset cifar100
#python score_networks.py --save_loc results/score/ImageNet16-120 --save_string all_score --valid --augtype none --repeat 1 --sigma 0.05 --nasspace nasbench201 --batch_size 128 --GPU 0 --dataset ImageNet16-120 --data_loc ../ImageNet16/

python score_networks.py --save_loc results/score/cifar10-test --save_string all_score --augtype none --repeat 1 --sigma 1 --nasspace nasbench201 --batch_size 128 --GPU 0 --n_samples 1000 --dataset cifar10 --data_loc ../cifardata --test
#python score_networks.py --save_loc results/score/cifar100-test --save_string all_score --augtype none --repeat 1 --sigma 0.05 --nasspace nasbench201 --batch_size 128 --GPU 0 --n_samples 1000 --dataset cifar100 --test
#python score_networks.py --save_loc results/score/ImageNet16-120-test --save_string all_score --augtype none --repeat 1 --sigma 0.05 --nasspace nasbench201 --batch_size 128 --GPU 0 --dataset ImageNet16-120 --data_loc ../ImageNet16/ --test

echo "Evaulation is Done."
