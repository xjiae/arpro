# #!/bin/bash

# # Array of MVTEC categories
# VISA_CATEGORIES=('candle' 'cashew' 'pipe_fryum' 'pcb1' 'pcb2' 'pcb3' 'pcb4' 'fryum' 'macaroni1' 'macaroni2'  'capsules' 'chewinggum' )

# # Number of GPUs
# NUM_GPUS=3

# # Function to find an empty GPU
# find_empty_gpu() {
#     for i in $(seq 0 $((NUM_GPUS - 1))); do
#         running_processes=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader)
#         gpu_uuid=$(nvidia-smi --query-gpu=gpu_uuid --format=csv,noheader | sed -n "$((i + 1))p" | xargs)
#         echo "Checking GPU $i with UUID $gpu_uuid..."
#         if ! echo "$running_processes" | grep -q "$gpu_uuid"; then
#             echo "GPU $i is available."
#             return $i
#         else
#             echo "GPU $i is currently in use."
#         fi
#     done
#     return -1
# }

# # Simple test to verify GPU utilization
# test_gpu() {
#     available_gpu=$1
#     echo "Running a simple test on GPU $available_gpu to verify utilization..."
#     CUDA_VISIBLE_DEVICES=$available_gpu python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); a = torch.randn((5000, 5000), device='cuda'); print(a.sum())"
# }

# # Loop through each category and run the training command on the next available GPU
# for category in "${VISA_CATEGORIES[@]}"; do
#     echo "Starting training for category: $category"
#     while true; do
#         find_empty_gpu
#         available_gpu=$?
#         if [ "$available_gpu" -ge 0 ]; then
#             echo "Using GPU $available_gpu for training category: $category"
#             test_gpu $available_gpu  # Simple GPU test to verify utilization
#             cmd="CUDA_VISIBLE_DEVICES=$available_gpu python train_ad.py --model efficientad --lr 0.0001 --epochs 500 --batch_size 8 --dataset visa --category $category --image_size 512"
#             echo "Running command: $cmd"
#             CUDA_VISIBLE_DEVICES=$available_gpu python train_ad.py --model efficientad --lr 0.0001 --epochs 500 --batch_size 8 --dataset visa --category $category --image_size 512
#             return_code=$?
#             echo "Return code: $return_code"
#             if [ $return_code -ne 0 ]; then
#                 echo "Error occurred during training for category: $category on GPU $available_gpu"
#                 exit 1
#             fi
#             echo "Finished training for category: $category on GPU $available_gpu"
#             break
#         else
#             echo "No available GPU found. Retrying in 60 seconds..."
#             sleep 60
#         fi
#     done
#     echo "Pausing for 10 seconds before starting the next category..."
#     sleep 10
# done

# echo "Training complete for all categories."


#!/bin/bash

# Array of categories 'pcb1' 'pcb2' 'pcb3' 'pcb4' 'fryum' 'macaroni1' 'macaroni2'  'capsules' 'chewinggum' 'pcb3' 
# VISA_CATEGORIES=('pcb4' 'fryum' 'macaroni1' 'macaroni2'  'capsules' 'chewinggum' )
VISA_CATEGORIES=('capsules' 'macaroni2')
# Loop through each category and run the training command
for category in "${VISA_CATEGORIES[@]}"
do
  echo "Starting evaluating for category: $category"
  CUDA_VISIBLE_DEVICES=0 python train_ad.py --model efficientad --lr 0.0001 --epochs 100 --batch_size 4 --dataset visa --image_size 512 --category $category
  # CUDA_VISIBLE_DEVICES=1 python train_fixer.py --model diffusion --dataset visa --category $category --image_size 512 --num_epochs 100 --batch_size 2
#    CUDA_VISIBLE_DEVICES=2 python run_eval.py --task exp2 --dataset visa --category $category --image_size 512 --batch_size 1 --noise 999 --end 10.0
  echo "Finished training for category: $category"
done

echo "Evaluation complete for all categories."
