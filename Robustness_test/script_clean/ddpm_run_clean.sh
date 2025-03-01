cd main_code

export GPU_IDS="6,7"
export NUM_TEST=128
export NUM_GPU=2
export BATCH_SIZE=22
export CLSSIFIER=t7
export DATASET="cifar10"

CUDA_VISIBLE_DEVICES=$GPU_IDS python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=30369 ddpm_attack_eval_clean.py --model_id ../../train/ddpm_ema_cifar10 --save_folder ddpm_clean_model --num_test $NUM_TEST  --batch_size $BATCH_SIZE --inference_step 50    --model_type $CLSSIFIER --fix_type large
CUDA_VISIBLE_DEVICES=$GPU_IDS python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=30369 ddpm_attack_eval_clean.py --model_id ../../train/ddpm_ema_cifar10 --save_folder ddpm_clean_model --num_test $NUM_TEST  --batch_size $BATCH_SIZE --inference_step 100   --model_type $CLSSIFIER --fix_type large
CUDA_VISIBLE_DEVICES=$GPU_IDS python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=30369 ddpm_attack_eval_clean.py --model_id ../../train/ddpm_ema_cifar10 --save_folder ddpm_clean_model --num_test $NUM_TEST  --batch_size $BATCH_SIZE --inference_step 200   --model_type $CLSSIFIER --fix_type large
CUDA_VISIBLE_DEVICES=$GPU_IDS python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=30369 ddpm_attack_eval_clean.py --model_id ../../train/ddpm_ema_cifar10 --save_folder ddpm_clean_model --num_test $NUM_TEST  --batch_size $BATCH_SIZE --inference_step 500   --model_type $CLSSIFIER --fix_type large
CUDA_VISIBLE_DEVICES=$GPU_IDS python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=30369 ddpm_attack_eval_clean.py --model_id ../../train/ddpm_ema_cifar10 --save_folder ddpm_clean_model --num_test $NUM_TEST  --batch_size $BATCH_SIZE --inference_step 1000  --model_type $CLSSIFIER --fix_type large



CUDA_VISIBLE_DEVICES=$GPU_IDS python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=30369 ddpm_attack_eval_clean.py --model_id ../../train/ddpm-cifar10-100e-ADDT --save_folder ddpm_cifar10_ADDT --num_test $NUM_TEST  --batch_size $BATCH_SIZE --inference_step 50    --model_type $CLSSIFIER --fix_type large --dataset $DATASET
CUDA_VISIBLE_DEVICES=$GPU_IDS python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=30369 ddpm_attack_eval_clean.py --model_id ../../train/ddpm-cifar10-100e-ADDT --save_folder ddpm_cifar10_ADDT --num_test $NUM_TEST  --batch_size $BATCH_SIZE --inference_step 100   --model_type $CLSSIFIER --fix_type large --dataset $DATASET
CUDA_VISIBLE_DEVICES=$GPU_IDS python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=30369 ddpm_attack_eval_clean.py --model_id ../../train/ddpm-cifar10-100e-ADDT --save_folder ddpm_cifar10_ADDT --num_test $NUM_TEST  --batch_size $BATCH_SIZE --inference_step 200   --model_type $CLSSIFIER --fix_type large --dataset $DATASET
CUDA_VISIBLE_DEVICES=$GPU_IDS python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=30369 ddpm_attack_eval_clean.py --model_id ../../train/ddpm-cifar10-100e-ADDT --save_folder ddpm_cifar10_ADDT --num_test $NUM_TEST  --batch_size $BATCH_SIZE --inference_step 500   --model_type $CLSSIFIER --fix_type large --dataset $DATASET
CUDA_VISIBLE_DEVICES=$GPU_IDS python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=30369 ddpm_attack_eval_clean.py --model_id ../../train/ddpm-cifar10-100e-ADDT --save_folder ddpm_cifar10_ADDT --num_test $NUM_TEST  --batch_size $BATCH_SIZE --inference_step 1000  --model_type $CLSSIFIER --fix_type large --dataset $DATASET

