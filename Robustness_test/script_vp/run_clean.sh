cd main_code

export GPU_IDS="0,1,2,3"
export NUM_TEST=1024
export NUM_GPU=4
export BATCH_SIZE=128
export CLSSIFIER=t7


CUDA_VISIBLE_DEVICES=$GPU_IDS python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=35369 vp_attack_eval_clean.py --model_id ../../train_vp/ddpmpp-cifar10-100e-fintune-ema --num_test $NUM_TEST  --batch_size $BATCH_SIZE --inference_step 1000  --model_type $CLSSIFIER --fix_type large
CUDA_VISIBLE_DEVICES=$GPU_IDS python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=35369 vp_attack_eval_clean.py --model_id ../../train_vp/ddpmpp-cifar10-100e-ADDT --num_test $NUM_TEST  --batch_size $BATCH_SIZE --inference_step 1000  --model_type $CLSSIFIER --fix_type large

