cd main_code

export GPU_IDS="0,1,2,3,4,5,6,7"
export NUM_TEST=1024
export NUM_GPU=8
export BATCH_SIZE=128
export CLSSIFIER=t7
export DATASET="cifar10"


CUDA_VISIBLE_DEVICES=$GPU_IDS python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=20369 edm_attack_eval_clean_vp.py  --save_folder vp_edm_clean_clean_model --num_test $NUM_TEST  --batch_size $BATCH_SIZE   --ckpt ../../edm_cifar_uncond_vp.pth   --model_type $CLSSIFIER  --sigma_max 0.5  --inference_step 50 --random_strength 6
CUDA_VISIBLE_DEVICES=$GPU_IDS python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=20369 edm_attack_eval_clean_vp.py  --save_folder vp_edm_clean_ADDT --num_test $NUM_TEST  --batch_size $BATCH_SIZE          --ckpt ../../ADDT-010000.pth           --model_type $CLSSIFIER  --sigma_max 0.5  --inference_step 50 --random_strength 6


