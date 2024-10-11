#!/bin/bash

# Array of MVTEC categories
MVTEC_CATEGORIES=("tile" "toothbrush" "wood" "zipper")

# Loop through each category and run the training command
for category in "${MVTEC_CATEGORIES[@]}"
do
  echo "Starting evaluating for category: $category"
  CUDA_VISIBLE_DEVICES=0 python train_ad.py --model efficientad --lr 0.0001 --epochs 100 --batch_size 4 --dataset mvtec --image_size 256 --category $category
  # CUDA_VISIBLE_DEVICES=1 python train_fixer.py --model diffusion --dataset visa --category $category --image_size 512 --num_epochs 100 --batch_size 2
#    CUDA_VISIBLE_DEVICES=2 python run_eval.py --task exp2 --dataset visa --category $category --image_size 512 --batch_size 1 --noise 999 --end 10.0
  echo "Finished training for category: $category"
done

echo "Evaluation complete for all categories."


# # Number of GPUs
# NUM_GPUS=3
# REQUIRED_MEMORY=5501  # Memory required per job in MiB

# # Function to find a GPU with sufficient available memory
# find_available_gpu() {
#     for i in $(seq 0 $((NUM_GPUS - 1))); do
#         free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sed -n "$((i + 1))p" | xargs)
#         echo "Checking GPU $i with $free_mem MiB free memory..."
#         if [ "$free_mem" -ge "$REQUIRED_MEMORY" ]; then
#             echo "GPU $i is available with $free_mem MiB free memory."
#             return $i
#         else
#             echo "GPU $i does not have enough free memory."
#         fi
#     done
#     return -1
# }

# # Loop through each category and run the training command on the next available GPU with sufficient memory
# for category in "${MVTEC_CATEGORIES[@]}"; do
#     echo "Starting training for category: $category"
#     while true; do
#         find_available_gpu
#         available_gpu=$?
#         if [ "$available_gpu" -ge 0 ]; then
#             echo "Using GPU $available_gpu for training category: $category"
#             cmd="CUDA_VISIBLE_DEVICES=$available_gpu python train_ad.py --model efficientad --lr 0.0001 --epochs 500 --batch_size 8 --dataset mvtec --category $category"
#             echo "Running command: $cmd"
#             CUDA_VISIBLE_DEVICES=$available_gpu python train_ad.py --model efficientad --lr 0.0001 --epochs 500 --batch_size 8 --dataset mvtec --category $category
#             return_code=$?
#             echo "Return code: $return_code"
#             if [ $return_code -ne 0 ]; then
#                 echo "Error occurred during training for category: $category on GPU $available_gpu"
#                 exit 1
#             fi
#             echo "Finished training for category: $category on GPU $available_gpu"
#             break
#         else
#             echo "No available GPU found with sufficient memory. Retrying in 60 seconds..."
#             sleep 60
#         fi
#     done
#     echo "Pausing for 10 seconds before starting the next category..."
#     sleep 10
# done

# echo "Training complete for all categories."
