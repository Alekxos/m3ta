export PYTHONPATH=$PYTHONPATH:$(pwd)

#CUDA_VISIBLE_DEVICES=0 python main.py --shared layer2 --rotation_type expand \
#			--group_norm 8 \
#			--nepoch 3 --milestone_1 1 --milestone_2 2 \
#			--outf /atlas/u/alekxos/domain_adaptation/mettta/ttt_cifar_release/results_meta/cifar10_layer2_gn_expand

CUDA_VISIBLE_DEVICES=0 python script_test_c10.py 5 layer2 online gn_expand
# CUDA_VISIBLE_DEVICES=0 python script_test_c10.py 5 layer2 slow gn_expand

#
#CUDA_VISIBLE_DEVICES=0 python script_test_c10.py 4 layer2 slow gn_expand
#CUDA_VISIBLE_DEVICES=0 python script_test_c10.py 4 layer2 online gn_expand
#
#CUDA_VISIBLE_DEVICES=0 python script_test_c10.py 3 layer2 slow gn_expand
#CUDA_VISIBLE_DEVICES=0 python script_test_c10.py 3 layer2 online gn_expand
#
#CUDA_VISIBLE_DEVICES=0 python script_test_c10.py 2 layer2 slow gn_expand
#CUDA_VISIBLE_DEVICES=0 python script_test_c10.py 2 layer2 online gn_expand
#
#CUDA_VISIBLE_DEVICES=0 python script_test_c10.py 1 layer2 slow gn_expand
#CUDA_VISIBLE_DEVICES=0 python script_test_c10.py 1 layer2 online gn_expand
#
#CUDA_VISIBLE_DEVICES=0 python script_test_c10.py 0 layer2 slow gn_expand
#CUDA_VISIBLE_DEVICES=0 python script_test_c10.py 0 layer2 online gn_expand
#
#
#CUDA_VISIBLE_DEVICES=0 python main.py --shared layer2 --rotation_type expand \
#			--nepoch 3 --milestone_1 1 --milestone_2 2 \
#			--outf /atlas/u/alekxos/domain_adaptation/mettta/ttt_cifar_release/results_meta/cifar10_layer2_bn_expand
#
#CUDA_VISIBLE_DEVICES=0 python script_test_c10.py 5 layer2 slow bn_expand
#CUDA_VISIBLE_DEVICES=0 python script_test_c10.py 5 layer2 online bn_expand