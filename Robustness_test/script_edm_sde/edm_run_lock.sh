cd main_code

export GPU_IDS="0,1,2,3,4,5,6,7"
export NUM_TEST=1024
export NUM_GPU=8
export BATCH_SIZE=128
export CLSSIFIER=t7
export DATASET="cifar10"


CUDA_VISIBLE_DEVICES=$GPU_IDS python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=30369 edm_attack_eval_linf_vp_lock.py  --save_folder vp_edm_linf_lock_clean_model --num_test $NUM_TEST  --batch_size $BATCH_SIZE  --model_type $CLSSIFIER  --ckpt ../../edm_cifar_uncond_vp.pth --sigma_max 0.5 --inference_step 50 --random_strength 6
CUDA_VISIBLE_DEVICES=$GPU_IDS python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=30369 edm_attack_eval_linf_vp_lock.py  --save_folder vp_edm_linf_lock_ADDT_model  --num_test $NUM_TEST  --batch_size $BATCH_SIZE  --model_type $CLSSIFIER  --ckpt ../../ADDT-010000.pth         --sigma_max 0.5 --inference_step 50 --random_strength 6