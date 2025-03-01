cd main_code

export GPU_IDS="4,5,6,7"
export NUM_TEST=1024
export NUM_GPU=4
export BATCH_SIZE=256
export CLSSIFIER=t7



CUDA_VISIBLE_DEVICES=$GPU_IDS python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=38369 ddpm_attack_eval_linf.py --model_id ../../train/ddpm_ema_cifar10 --save_folder ddpm_clean_model_linf --num_test $NUM_TEST  --batch_size $BATCH_SIZE --inference_step 50    --model_type $CLSSIFIER --fix_type large
CUDA_VISIBLE_DEVICES=$GPU_IDS python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=38369 ddpm_attack_eval_linf.py --model_id ../../train/ddpm_ema_cifar10 --save_folder ddpm_clean_model_linf --num_test $NUM_TEST  --batch_size $BATCH_SIZE --inference_step 100   --model_type $CLSSIFIER --fix_type large
CUDA_VISIBLE_DEVICES=$GPU_IDS python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=38369 ddpm_attack_eval_linf.py --model_id ../../train/ddpm_ema_cifar10 --save_folder ddpm_clean_model_linf --num_test $NUM_TEST  --batch_size $BATCH_SIZE --inference_step 200   --model_type $CLSSIFIER --fix_type large
CUDA_VISIBLE_DEVICES=$GPU_IDS python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=38369 ddpm_attack_eval_linf.py --model_id ../../train/ddpm_ema_cifar10 --save_folder ddpm_clean_model_linf --num_test $NUM_TEST  --batch_size $BATCH_SIZE --inference_step 500   --model_type $CLSSIFIER --fix_type large
CUDA_VISIBLE_DEVICES=$GPU_IDS python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=35369 ddpm_attack_eval_linf.py --model_id ../../train/ddpm_ema_cifar10 --save_folder ddpm_clean_model_linf --num_test $NUM_TEST  --batch_size $BATCH_SIZE --inference_step 1000  --model_type $CLSSIFIER --fix_type large


CUDA_VISIBLE_DEVICES=$GPU_IDS python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=38369 ddpm_attack_eval_linf.py --model_id ../../train/ddpm-cifar10-100e-ADDT --save_folder ddpm_ADDT_linf --num_test $NUM_TEST  --batch_size $BATCH_SIZE --inference_step 50    --model_type $CLSSIFIER --fix_type large
CUDA_VISIBLE_DEVICES=$GPU_IDS python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=38369 ddpm_attack_eval_linf.py --model_id ../../train/ddpm-cifar10-100e-ADDT --save_folder ddpm_ADDT_linf --num_test $NUM_TEST  --batch_size $BATCH_SIZE --inference_step 100   --model_type $CLSSIFIER --fix_type large
CUDA_VISIBLE_DEVICES=$GPU_IDS python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=38369 ddpm_attack_eval_linf.py --model_id ../../train/ddpm-cifar10-100e-ADDT --save_folder ddpm_ADDT_linf --num_test $NUM_TEST  --batch_size $BATCH_SIZE --inference_step 200   --model_type $CLSSIFIER --fix_type large
CUDA_VISIBLE_DEVICES=$GPU_IDS python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=38369 ddpm_attack_eval_linf.py --model_id ../../train/ddpm-cifar10-100e-ADDT --save_folder ddpm_ADDT_linf --num_test $NUM_TEST  --batch_size $BATCH_SIZE --inference_step 500   --model_type $CLSSIFIER --fix_type large
CUDA_VISIBLE_DEVICES=$GPU_IDS python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=35369 ddpm_attack_eval_linf.py --model_id ../../train/ddpm-cifar10-100e-ADDT --save_folder ddpm_ADDT_linf --num_test $NUM_TEST  --batch_size $BATCH_SIZE --inference_step 1000  --model_type $CLSSIFIER --fix_type large