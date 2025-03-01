cd main_code

export GPU_IDS="0,1,2,3"
export NUM_TEST=1024
export NUM_GPU=4
export BATCH_SIZE=128
export CLSSIFIER=t7


CUDA_VISIBLE_DEVICES=$GPU_IDS python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=32469 ddim_attack_eval_linf.py --model_id ../../train/ddpm_ema_cifar10 --save_folder ddim_clean_model_linf --num_test $NUM_TEST  --batch_size $BATCH_SIZE --inference_step 50    --model_type $CLSSIFIER 
CUDA_VISIBLE_DEVICES=$GPU_IDS python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=32469 ddim_attack_eval_linf.py --model_id ../../train/ddpm_ema_cifar10 --save_folder ddim_clean_model_linf --num_test $NUM_TEST  --batch_size $BATCH_SIZE --inference_step 100   --model_type $CLSSIFIER 
CUDA_VISIBLE_DEVICES=$GPU_IDS python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=32469 ddim_attack_eval_linf.py --model_id ../../train/ddpm_ema_cifar10 --save_folder ddim_clean_model_linf --num_test $NUM_TEST  --batch_size $BATCH_SIZE --inference_step 200   --model_type $CLSSIFIER 
CUDA_VISIBLE_DEVICES=$GPU_IDS python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=32469 ddim_attack_eval_linf.py --model_id ../../train/ddpm_ema_cifar10 --save_folder ddim_clean_model_linf --num_test $NUM_TEST  --batch_size $BATCH_SIZE --inference_step 500   --model_type $CLSSIFIER 
CUDA_VISIBLE_DEVICES=$GPU_IDS python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=32469 ddim_attack_eval_linf.py --model_id ../../train/ddpm_ema_cifar10 --save_folder ddim_clean_model_linf --num_test $NUM_TEST  --batch_size $BATCH_SIZE --inference_step 1000  --model_type $CLSSIFIER 

CUDA_VISIBLE_DEVICES=$GPU_IDS python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=32469 ddim_attack_eval_linf.py --model_id ../../train/ddpm-cifar10-100e-ADDT --save_folder ddim_ADDT_linf --num_test $NUM_TEST  --batch_size $BATCH_SIZE --inference_step 50    --model_type $CLSSIFIER 
CUDA_VISIBLE_DEVICES=$GPU_IDS python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=32469 ddim_attack_eval_linf.py --model_id ../../train/ddpm-cifar10-100e-ADDT --save_folder ddim_ADDT_linf --num_test $NUM_TEST  --batch_size $BATCH_SIZE --inference_step 100   --model_type $CLSSIFIER 
CUDA_VISIBLE_DEVICES=$GPU_IDS python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=32469 ddim_attack_eval_linf.py --model_id ../../train/ddpm-cifar10-100e-ADDT --save_folder ddim_ADDT_linf --num_test $NUM_TEST  --batch_size $BATCH_SIZE --inference_step 200   --model_type $CLSSIFIER 
CUDA_VISIBLE_DEVICES=$GPU_IDS python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=32469 ddim_attack_eval_linf.py --model_id ../../train/ddpm-cifar10-100e-ADDT --save_folder ddim_ADDT_linf --num_test $NUM_TEST  --batch_size $BATCH_SIZE --inference_step 500   --model_type $CLSSIFIER 
CUDA_VISIBLE_DEVICES=$GPU_IDS python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=32469 ddim_attack_eval_linf.py --model_id ../../train/ddpm-cifar10-100e-ADDT --save_folder ddim_ADDT_linf --num_test $NUM_TEST  --batch_size $BATCH_SIZE --inference_step 1000  --model_type $CLSSIFIER 