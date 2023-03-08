python score_networks.py --save_loc $1 --save_string all_score --trainval --augtype none --repeat 1 --sigma 0.05 --nasspace nasbench201 --batch_size 128 --GPU 0 --n_samples 50 --dataset cifar10

<<plot_example
python process_data.py --save_loc $1 --oper ninaswot_add_ntk_add_entropy --save_string all_score --trainval --augtype none --repeat 1 --sigma 0.05 --nasspace nasbench201 --batch_size 128 --GPU 0 --n_samples 50 --dataset cifar10
python plot_2d_scores.py --save_loc $1 --targets acc-ninaswot_add_ntk_add_entropy --save_string all_score --trainval --augtype none --repeat 1 --sigma 0.05 --nasspace nasbench201 --batch_size 128 --GPU 0 --n_samples 50 --dataset cifar10
python plot_3d_scores.py --save_loc $1 --save_string all_score --trainval --augtype none --repeat 1 --sigma 0.05 --nasspace nasbench201 --batch_size 128 --GPU 0 --n_samples 50 --dataset cifar10
plot_example

#python score_networks.py --save_loc $1 --save_string all_score --trainval --augtype none --repeat 1 --score 1_-1_1 --sigma 0.05 --nasspace nasbench201 --batch_size 128 --GPU 0 --dataset cifar100 --data_loc ../cifar100/
#python score_networks.py --save_loc $1 --save_string all_score --trainval --augtype none --repeat 1 --score 1_-1_1 --sigma 0.05 --nasspace nasbench201 --batch_size 128 --GPU 0 --dataset ImageNet16-120 --data_loc ../ImageNet16/

<<block
python score_networks.py --trainval --augtype none --repeat 1 --score hook_logdet --sigma 0.05 --nasspace nds_pnas --batch_size 128 --GPU 3
python score_networks.py --trainval --augtype none --repeat 1 --score hook_logdet --sigma 0.05 --nasspace nds_enas --batch_size 128 --GPU 3
python score_networks.py --trainval --augtype none --repeat 1 --score hook_logdet --sigma 0.05 --nasspace nds_darts --batch_size 128 --GPU 3
python score_networks.py --trainval --augtype none --repeat 1 --score hook_logdet --sigma 0.05 --nasspace nds_darts_fix-w-d --batch_size 128 --GPU 3
python score_networks.py --trainval --augtype none --repeat 1 --score hook_logdet --sigma 0.05 --nasspace nds_nasnet --batch_size 128 --GPU 3
python score_networks.py --trainval --augtype none --repeat 1 --score hook_logdet --sigma 0.05 --nasspace nds_amoeba --batch_size 128 --GPU 3
python score_networks.py --trainval --augtype none --repeat 1 --score hook_logdet --sigma 0.05 --nasspace nds_resnet --batch_size 128 --GPU 3
python score_networks.py --trainval --augtype none --repeat 1 --score hook_logdet --sigma 0.05 --nasspace nds_resnext-a --batch_size 128 --GPU 3
python score_networks.py --trainval --augtype none --repeat 1 --score hook_logdet --sigma 0.05 --nasspace nds_resnext-b --batch_size 128 --GPU 3



python score_networks.py --trainval --augtype none --repeat 1 --score hook_logdet --sigma 0.05 --nasspace amoeba_in --batch_size 128 --GPU 3 --dataset imagenette2 --data_loc ../imagenette2/
python score_networks.py --trainval --augtype none --repeat 1 --score hook_logdet --sigma 0.05 --nasspace nds_amoeba_in --batch_size 128 --GPU 3 --dataset imagenette2 --data_loc ../imagenette2/
python score_networks.py --trainval --augtype none --repeat 1 --score hook_logdet --sigma 0.05 --nasspace nds_darts_in --batch_size 128 --GPU 3 --dataset imagenette2 --data_loc ../imagenette2/
python score_networks.py --trainval --augtype none --repeat 1 --score hook_logdet --sigma 0.05 --nasspace nds_nasnet_in --batch_size 128 --GPU 3 --dataset imagenette2 --data_loc ../imagenette2/
python score_networks.py --trainval --augtype none --repeat 1 --score hook_logdet --sigma 0.05 --nasspace nds_pnas_in --batch_size 128 --GPU 3 --dataset imagenette2 --data_loc ../imagenette2/
python score_networks.py --trainval --augtype none --repeat 1 --score hook_logdet --sigma 0.05 --nasspace nds_enas_in --batch_size 128 --GPU 3 --dataset imagenette2 --data_loc ../imagenette2/
python score_networks.py --trainval --augtype none --repeat 1 --score hook_logdet --sigma 0.05 --nasspace nds_resnext-a_in --batch_size 128 --GPU 3 --dataset imagenette2 --data_loc ../imagenette2/



python score_networks.py --trainval --augtype none --repeat 1 --score hook_logdet --sigma 0.05 --nasspace nasbench201 --batch_size 128 --GPU 3 --dataset cifar100 --data_loc ../cifar100/
python score_networks.py --trainval --augtype none --repeat 1 --score hook_logdet --sigma 0.05 --nasspace nasbench201 --batch_size 128 --GPU 3 --dataset ImageNet16-120 --data_loc ../imagenet16/Imagenet16/

python score_networks.py --trainval --augtype none --repeat 1 --score hook_logdet --sigma 0.05 --nasspace nasbench101 --batch_size 128 --GPU 3 --api_loc ../nasbench_only108.tfrecord 
block

echo "Evaulation is Done."
